use crate::hir::infer::PrincipalShape;
use crate::hir::types::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::time::Duration;

static Z3_WARNED: OnceLock<bool> = OnceLock::new();

/// Default timeout for Z3 solver invocations (milliseconds).
pub(crate) const Z3_TIMEOUT_MS: u64 = 5_000;
/// Longer timeout for the independent BII template verifier
/// (`verify_template_against_problem`): verification is the last line of
/// defense after synthesis, so it gets a longer budget than synthesis
/// queries (committee ruling on the verifier-overhead decision point).
pub(crate) const VERIFY_TIMEOUT_MS: u64 = 60_000;
/// Default memory limit for Z3 (megabytes).
const Z3_MEMORY_LIMIT_MB: u64 = 512;
/// Minimum required Z3 major version.
const Z3_MIN_VERSION: &str = "4.8.0";

/// SMT-LIB2-based unicity checker using Z3.
///
/// Encodes the constraint context `C` as first-order formulas over an
/// uninterpreted sort `Type`, then queries Z3 for the unique shape of a
/// target variable (O'Brien, Rémy & Scherer §4.1):
///
///   C[τ!ζ] iff ∀φ, φ ⊢ [C[τ = g]] ⇒ shape(g) = ζ
///
/// Z3 is resolved via `PATH` by default. To bundle Z3 into the final
/// binary, add `z3 = { version = "0.20.2", features = ["vendored"] }`
/// to Cargo.toml and replace this module's internals with the z3 crate API.
/// RAII guard that kills the z3 child on drop — the cargo/rustc
/// kill-on-drop pattern, written by hand because this toolchain's std
/// lacks `Command::kill_on_drop`.  The child is reaped on every
/// non-SIGINT exit path (stdin write failure, wait failure, early return
/// via `?`, panic unwind).  On SIGINT the compiler process dies without
/// running Drop, so the child can still outlive it — but z3 is bounded by
/// the `:timeout`/`:memory_max_size` options set on the query, so the
/// orphan exits on its own.
struct KillOnDropChild(std::process::Child);

impl KillOnDropChild {
    /// Equivalent of `Child::wait_with_output` (reads the piped
    /// stdout/stderr, then waits), but keeps the kill-on-drop guarantee:
    /// std's method consumes `self`, which would bypass the guard's Drop
    /// on the error path, so we reimplement it here.
    fn wait_with_output(mut self) -> std::io::Result<std::process::Output> {
        use std::io::Read;
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        // Read BOTH pipes CONCURRENTLY: the sequential version (stdout to
        // completion, then stderr) deadlocks when the child fills the
        // stderr pipe buffer (~64 KB) while the parent is still draining
        // stdout — the child blocks on the stderr write, the parent
        // blocks on the stdout read.  (The standard `Child::wait_with_output`
        // does the same; it cannot be used here because it consumes `self`
        // and would bypass `KillOnDropChild::drop`.)
        std::thread::scope(|s| {
            let so = self.0.stdout.take();
            let se = self.0.stderr.take();
            let h1 = s.spawn(|| {
                if let Some(mut so) = so {
                    let _ = so.read_to_end(&mut stdout);
                }
            });
            let h2 = s.spawn(|| {
                if let Some(mut se) = se {
                    let _ = se.read_to_end(&mut stderr);
                }
            });
            let _ = h1.join();
            let _ = h2.join();
        });
        let status = self.0.wait()?;
        Ok(std::process::Output {
            status,
            stdout,
            stderr,
        })
    }
}

impl Drop for KillOnDropChild {
    fn drop(&mut self) {
        let _ = self.0.kill();
    }
}

pub struct SmtSolver {
    solver_path: String,
    /// Cache of SMT query → result, avoiding re-spawning Z3 for identical queries.
    query_cache: RefCell<HashMap<String, SmtResult>>,
    /// Per-instance solver timeout in milliseconds (overrides the default
    /// `Z3_TIMEOUT_MS` for callers that need heavier quantified queries).
    /// Interior mutability so `set_timeout` can re-tune a shared instance
    /// in place (e.g. the verifier gets a longer budget than synthesis).
    timeout_ms: RefCell<u64>,
}

impl SmtSolver {
    pub fn new(solver_path: &str) -> Self {
        SmtSolver::with_timeout(solver_path, Z3_TIMEOUT_MS)
    }

    /// Construct a solver with an explicit per-query timeout (milliseconds).
    /// BII synthesis tests with wide templates (e.g. 3-variable support-
    /// three rows) exceed the 5 s default on their quantified ∃∀ queries.
    pub fn with_timeout(solver_path: &str, timeout_ms: u64) -> Self {
        let solver = SmtSolver {
            solver_path: solver_path.to_string(),
            query_cache: RefCell::new(HashMap::new()),
            timeout_ms: RefCell::new(timeout_ms),
        };
        // Verify Z3 version on first use (lazy, via check_version).
        solver
    }

    /// Adjust the per-query timeout at runtime. Interior mutability keeps
    /// the `&self` API so a shared solver instance can be re-tuned in
    /// place (the BII verifier gets `VERIFY_TIMEOUT_MS` while synthesis
    /// keeps the default).
    pub(crate) fn set_timeout(&self, timeout_ms: u64) {
        *self.timeout_ms.borrow_mut() = timeout_ms;
    }

    /// Verify that the Z3 binary meets the minimum version requirement.
    /// Returns `true` if the version check passes or if Z3 is not found (warning only).
    pub fn check_version(&self) -> bool {
        let output = match Command::new(&self.solver_path)
            .arg("--version")
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .output()
        {
            Ok(o) if o.status.success() => o,
            _ => return false,
        };
        let stdout = String::from_utf8_lossy(&output.stdout);
        // Z3 --version outputs: "Z3 version 4.8.12 - 64 bit"
        let version_str = match stdout.split_whitespace().nth(2) {
            Some(v) => v,
            None => return false,
        };
        let parts: Vec<u64> = version_str
            .split('.')
            .filter_map(|p| p.parse::<u64>().ok())
            .collect();
        if parts.len() < 2 {
            return false;
        }
        let min_parts: Vec<u64> = Z3_MIN_VERSION
            .split('.')
            .filter_map(|p| p.parse::<u64>().ok())
            .collect();
        for (i, &p) in parts.iter().enumerate() {
            let min = min_parts.get(i).copied().unwrap_or(0);
            if p < min {
                return false;
            }
            if p > min {
                return true;
            }
        }
        true
    }

    /// Main entry: check whether `ty` (an InferVar or resolved type) has a
    /// unique shape given the constraint context.
    ///
    /// `bindings` maps InferVar ids → resolved concrete `TypeId`.
    /// `eq_constraints` is a set of InferVar–InferVar equality pairs.
    pub fn check_unicity<'input>(
        &self,
        ctx: &TypeContext<'input>,
        ty: TypeId,
        bindings: &HashMap<usize, TypeId>,
        eq_constraints: &[(usize, usize)],
    ) -> Option<PrincipalShape> {
        let resolved = ctx.resolve_binding(ty);

        // If already concrete, shape is known immediately.
        if !matches!(ctx.get(resolved), TypeData::InferVar { .. }) {
            return Some(match ctx.get(resolved) {
                TypeData::Fn { .. } => PrincipalShape::Arrow,
                TypeData::Tuple { elems } => PrincipalShape::Tuple(elems.len()),
                TypeData::Adt { args, .. } => PrincipalShape::Constructor(args.len()),
                TypeData::Forall { .. }
                | TypeData::Exists { .. }
                | TypeData::Poly { .. }
                | TypeData::SkolemVar { .. } => PrincipalShape::Poly,
                TypeData::Int { .. }
                | TypeData::UInt { .. }
                | TypeData::Float { .. }
                | TypeData::Bool
                | TypeData::Char
                | TypeData::Byte
                | TypeData::USize
                | TypeData::Rational { .. } => PrincipalShape::Scalar,
                _ => PrincipalShape::Unknown,
            });
        }

        let var_id = match ctx.get(resolved) {
            TypeData::InferVar { id, .. } => *id,
            _ => return None,
        };

        // Build the full SMT-LIB2 query
        let mut smt = String::new();
        smt.push_str("(set-option :produce-models true)\n");
        smt.push_str("(set-logic ALL)\n\n");

        // ── 1. Declare uninterpreted sort Type ────────────────
        smt.push_str("(declare-sort Type 0)\n\n");

        // ── 2. Shape tag constants ──────────────────────────────
        smt.push_str("(declare-const SHAPE_UNKNOWN Int)\n");
        smt.push_str("(declare-const SHAPE_ARROW Int)\n");
        smt.push_str("(declare-const SHAPE_TUPLE Int)\n");
        smt.push_str("(declare-const SHAPE_CONSTRUCTOR Int)\n");
        smt.push_str("(declare-const SHAPE_POLY Int)\n");
        smt.push_str("(declare-const SHAPE_SCALAR Int)\n");
        smt.push_str("(assert (= SHAPE_UNKNOWN 0))\n");
        smt.push_str("(assert (= SHAPE_ARROW 1))\n");
        smt.push_str("(assert (= SHAPE_TUPLE 2))\n");
        smt.push_str("(assert (= SHAPE_CONSTRUCTOR 3))\n");
        smt.push_str("(assert (= SHAPE_POLY 4))\n");
        smt.push_str("(assert (= SHAPE_SCALAR 5))\n\n");

        // ── 3. Type constructor functions ────────────────────────
        smt.push_str("(declare-fun type-int32 () Type)\n");
        smt.push_str("(declare-fun type-int64 () Type)\n");
        smt.push_str("(declare-fun type-bool () Type)\n");
        smt.push_str("(declare-fun type-unit () Type)\n");
        smt.push_str("(declare-fun type-never () Type)\n");
        smt.push_str("(declare-fun type-char () Type)\n");
        smt.push_str("(declare-fun type-byte () Type)\n");
        smt.push_str("(declare-fun type-fn (Type Type) Type)\n");
        smt.push_str("(declare-fun type-tuple2 (Type Type) Type)\n");
        smt.push_str("(declare-fun type-struct (Int Type) Type)\n");
        smt.push_str("(declare-fun type-poly (Type) Type)\n");
        smt.push_str("(declare-fun type-rational (Int Int) Type)\n");
        smt.push_str("(declare-fun type-ref (Type Bool) Type)\n");
        smt.push_str("(declare-fun type-ptr (Type Type) Type)\n");
        smt.push_str("(declare-fun type-slice (Type) Type)\n");
        smt.push_str("(declare-fun type-array (Type Int) Type)\n");
        smt.push_str("(declare-fun type-coproduct (Type Type) Type)\n");
        smt.push_str("(declare-fun type-pointer (Type) Type)\n");
        smt.push_str("(declare-fun type-float32 () Type)\n");
        smt.push_str("(declare-fun type-float64 () Type)\n");
        smt.push_str("(declare-fun type-dyn-trait (Int) Type)\n");
        smt.push_str("(declare-fun type-type () Type)\n\n");

        // ── 4. Shape-of and arity-of functions ────────────────────
        smt.push_str("(declare-fun shape-of (Type) Int)\n");
        smt.push_str("(declare-fun arity-of (Type) Int)\n\n");

        // ── 5. Shape and arity axioms ────────────────────────────
        smt.push_str("(assert (= (shape-of type-int32) SHAPE_SCALAR))\n");
        smt.push_str("(assert (= (shape-of type-int64) SHAPE_SCALAR))\n");
        smt.push_str("(assert (= (shape-of type-bool) SHAPE_SCALAR))\n");
        smt.push_str("(assert (= (shape-of type-unit) SHAPE_UNKNOWN))\n");
        smt.push_str("(assert (= (shape-of type-never) SHAPE_UNKNOWN))\n");
        smt.push_str("(assert (= (shape-of type-char) SHAPE_SCALAR))\n");
        smt.push_str("(assert (= (shape-of type-byte) SHAPE_SCALAR))\n");
        smt.push_str("(assert (= (shape-of type-float32) SHAPE_SCALAR))\n");
        smt.push_str("(assert (= (shape-of type-float64) SHAPE_SCALAR))\n");
        smt.push_str(
            "(assert (forall ((a Type) (b Type)) (and (= (shape-of (type-fn a b)) SHAPE_ARROW) (= (arity-of (type-fn a b)) 2))))\n",
        );
        smt.push_str(
            "(assert (forall ((a Type) (b Type)) (and (= (shape-of (type-tuple2 a b)) SHAPE_TUPLE) (= (arity-of (type-tuple2 a b)) 2))))\n",
        );
        smt.push_str("(assert (forall ((tag Int) (a Type)) (and (= (shape-of (type-struct tag a)) SHAPE_CONSTRUCTOR) (= (arity-of (type-struct tag a)) 1))))\n");
        smt.push_str("(assert (forall ((a Type)) (= (shape-of (type-poly a)) SHAPE_POLY)))\n");
        smt.push_str(
            "(assert (forall ((p Int) (q Int)) (= (shape-of (type-rational p q)) SHAPE_SCALAR)))\n",
        );
        smt.push_str("(assert (forall ((a Type) (m Bool)) (and (= (shape-of (type-ref a m)) SHAPE_CONSTRUCTOR) (= (arity-of (type-ref a m)) 1))))\n");
        smt.push_str("(assert (forall ((s Type) (p Type)) (and (= (shape-of (type-ptr s p)) SHAPE_CONSTRUCTOR) (= (arity-of (type-ptr s p)) 2))))\n");
        smt.push_str("(assert (forall ((a Type)) (and (= (shape-of (type-slice a)) SHAPE_CONSTRUCTOR) (= (arity-of (type-slice a)) 1))))\n");
        smt.push_str("(assert (forall ((a Type) (n Int)) (and (= (shape-of (type-array a n)) SHAPE_CONSTRUCTOR) (= (arity-of (type-array a n)) 1))))\n");
        smt.push_str("(assert (forall ((a Type) (b Type)) (and (= (shape-of (type-coproduct a b)) SHAPE_CONSTRUCTOR) (= (arity-of (type-coproduct a b)) 2))))\n");
        smt.push_str("(assert (forall ((a Type)) (and (= (shape-of (type-pointer a)) SHAPE_CONSTRUCTOR) (= (arity-of (type-pointer a)) 1))))\n");
        smt.push_str(
            "(assert (forall ((tag Int)) (= (shape-of (type-dyn-trait tag)) SHAPE_CONSTRUCTOR)))\n",
        );
        smt.push_str("(assert (= (shape-of type-type) SHAPE_UNKNOWN))\n\n");

        // ── 6. Inference variable ──────────────────────────────
        smt.push_str(&format!("(declare-const iv_{} Type)\n", var_id));

        // ── 7. Bindings ──────────────────────────────────────────
        for (vid, bound_ty) in bindings {
            let term = Self::type_to_smt_term(ctx, *bound_ty);
            smt.push_str(&format!("(assert (= iv_{} {}))\n", vid, term));
        }

        // ── 8. Eq constraints ────────────────────────────────────
        for (a, b) in eq_constraints {
            smt.push_str(&format!("(assert (= iv_{} iv_{}))\n", a, b));
        }

        // ── 9. Push/assert/pop for each candidate shape ──────────
        let shape_names = [
            ("SHAPE_UNKNOWN", PrincipalShape::Unknown),
            ("SHAPE_SCALAR", PrincipalShape::Scalar),
            ("SHAPE_ARROW", PrincipalShape::Arrow),
            ("SHAPE_TUPLE", PrincipalShape::Tuple(2)),
            ("SHAPE_CONSTRUCTOR", PrincipalShape::Constructor(0)),
            ("SHAPE_POLY", PrincipalShape::Poly),
        ];

        for (name, _ps) in &shape_names {
            smt.push_str(&format!(
                "(push 1)\n\
                 (assert (= (shape-of iv_{}) {}))\n\
                 (check-sat)\n\
                 (pop 1)\n",
                var_id, name
            ));
        }

        // ── 10. Query Z3 ─────────────────────────────────────────
        let output = self.call_z3(&smt);
        match output {
            SmtResult::Sat(result) => Self::parse_unicity_results(&result, &shape_names),
            SmtResult::Unsat => {
                // Unsat: Z3 proved that no model satisfies the constraints.
                // The target variable has no possible shape, which means
                // the constraints are contradictory.  We cannot determine
                // a unique shape — return None.
                None
            }
            SmtResult::Unknown => {
                // Z3 answered `unknown` (undecided) — fail closed: no
                // shape is asserted, matching the timeout/error fallback.
                None
            }
            SmtResult::Timeout => {
                // Timed out — conservatively return Unknown to avoid false positives.
                Some(PrincipalShape::Unknown)
            }
            SmtResult::Error(_) => {
                // Z3 not available or error — fall back to heuristic.
                Some(PrincipalShape::Unknown)
            }
        }
    }

    /// Convert a TypeId to an SMT-LIB2 term.
    fn type_to_smt_term<'input>(ctx: &TypeContext<'input>, ty: TypeId) -> String {
        let resolved = ctx.resolve_binding(ty);
        match ctx.get(resolved) {
            TypeData::Int { bits, .. } => {
                if *bits == 32 {
                    "type-int32".into()
                } else {
                    "type-int64".into()
                }
            }
            TypeData::UInt { .. } => "type-int64".into(),
            TypeData::Bool => "type-bool".into(),
            TypeData::Unit => "type-unit".into(),
            TypeData::Never => "type-never".into(),
            TypeData::Char => "type-char".into(),
            TypeData::Byte => "type-byte".into(),
            TypeData::Fn { params, ret } => {
                if params.len() == 1 {
                    let p = Self::type_to_smt_term(ctx, params[0]);
                    let r = Self::type_to_smt_term(ctx, *ret);
                    format!("(type-fn {} {})", p, r)
                } else if params.len() == 2 {
                    let p1 = Self::type_to_smt_term(ctx, params[0]);
                    let p2 = Self::type_to_smt_term(ctx, params[1]);
                    let r = Self::type_to_smt_term(ctx, *ret);
                    format!("(type-fn {} (type-fn {} {}))", p1, p2, r)
                } else {
                    "type-unknown".into()
                }
            }
            TypeData::Tuple { elems } => {
                if elems.is_empty() {
                    "type-unit".into()
                } else if elems.len() == 1 {
                    Self::type_to_smt_term(ctx, elems[0])
                } else {
                    let a = Self::type_to_smt_term(ctx, elems[0]);
                    let b = Self::type_to_smt_term(ctx, elems[1]);
                    format!("(type-tuple2 {} {})", a, b)
                }
            }
            TypeData::Forall { body, .. }
            | TypeData::Exists { base: body, .. }
            | TypeData::Poly { body, .. }
            | TypeData::Mu { body, .. }
            | TypeData::Nu { body, .. } => {
                let b = Self::type_to_smt_term(ctx, *body);
                format!("(type-poly {})", b)
            }
            TypeData::InferVar { id, .. } => format!("iv_{}", id),
            TypeData::Rational {
                int_bits,
                frac_bits,
            } => {
                format!("(type-rational {} {})", int_bits, frac_bits)
            }
            TypeData::USize => "type-int64".into(),
            TypeData::Ref { ty, mutable, .. } => {
                let inner = Self::type_to_smt_term(ctx, *ty);
                let m = if *mutable { "true" } else { "false" };
                format!("(type-ref {} {})", inner, m)
            }
            TypeData::Pointer { ty } => {
                let inner = Self::type_to_smt_term(ctx, *ty);
                format!("(type-pointer {})", inner)
            }
            TypeData::Ptr { size, pointee } => {
                let s = Self::type_to_smt_term(ctx, *size);
                let p = Self::type_to_smt_term(ctx, *pointee);
                format!("(type-ptr {} {})", s, p)
            }
            TypeData::Slice { elem } => {
                let e = Self::type_to_smt_term(ctx, *elem);
                format!("(type-slice {})", e)
            }
            TypeData::Array { elem, size } => {
                let e = Self::type_to_smt_term(ctx, *elem);
                format!("(type-array {} {})", e, size)
            }
            TypeData::Coproduct { alternatives } => {
                if alternatives.len() == 2 {
                    let a = Self::type_to_smt_term(ctx, alternatives[0]);
                    let b = Self::type_to_smt_term(ctx, alternatives[1]);
                    format!("(type-coproduct {} {})", a, b)
                } else if alternatives.len() == 1 {
                    Self::type_to_smt_term(ctx, alternatives[0])
                } else {
                    "type-unknown".into()
                }
            }
            TypeData::Adt {
                kind: _,
                def_id,
                args,
            } => {
                // Encode as (type-struct def_id first_arg) for the first arg
                if let Some(&arg) = args.first() {
                    let a = Self::type_to_smt_term(ctx, arg);
                    format!("(type-struct {} {})", def_id.0, a)
                } else {
                    format!("(type-struct {} type-unit)", def_id.0)
                }
            }
            TypeData::DynTrait { traits } => {
                let tag = traits.first().map(|t| t.0 as i64).unwrap_or(0);
                format!("(type-dyn-trait {})", tag)
            }
            TypeData::AssociatedType { self_ty, .. } => Self::type_to_smt_term(ctx, *self_ty),
            TypeData::Error => "type-unknown".into(),
            TypeData::GenericParam { .. } => "type-unknown".into(),
            TypeData::Float { bits } => {
                if *bits == 32 {
                    "type-float32".into()
                } else {
                    "type-float64".into()
                }
            }
            TypeData::SkolemVar { .. } => "type-unknown".to_string(),
            // Regex types cannot appear in contracts (SYNTAX.md §Compile-Time Regular Expressions).
            // SMT solvers have limited string theory, so we don't encode them.
            TypeData::Regex { .. } => "type-unknown".into(),
            TypeData::Type => "type-type".into(),
            TypeData::Opaque { .. } => "type-unknown".into(),
        }
    }
}

/// Result of an SMT query.
#[derive(Debug, Clone, PartialEq)]
pub enum SmtResult {
    /// Z3 returned sat/unsat successfully.
    Sat(String),
    Unsat,
    /// Z3 returned `unknown` — the query is undecided (e.g. quantified
    /// formulas beyond Z3's reasoning).  Never treat this as sat/unsat;
    /// the caller must fail closed.
    Unknown,
    /// Z3 timed out — the query could not be resolved within the budget.
    /// The caller should fall back to a conservative (safe) heuristic.
    Timeout,
    /// Z3 could not be started or the query failed for other reasons.
    Error(String),
}

/// Explicit tri-state outcome of a caller-built raw SMT query, used by the
/// BII synthesis `Refine` step.  `Unknown` (undecided, timeout) and
/// `Error` (solver unavailable) are distinct from a decisive `Sat`/`Unsat`
/// — the caller fails closed on anything non-decisive.
#[derive(Debug, Clone, PartialEq)]
pub enum RawQueryOutcome {
    /// Z3 answered `sat`; the payload is the raw stdout (which may include
    /// a `(get-model)` block when the caller requested one).
    Sat(String),
    /// Z3 answered `unsat`.
    Unsat,
    /// Z3 answered `unknown` or timed out — undecided, fail closed.
    Unknown,
    /// Z3 could not be started or the query errored — fail closed.
    Error(String),
}

/// Translate a Posita invariant expression into an SMT-LIB2 term (linear
/// integer arithmetic).  Returns `false` for shapes the translator does
/// not support (calls, indexing, division, bit-ops, ...) — the caller then
/// drops the candidate (fail-closed).  `pub(crate)`: also used by the
/// checker's loop-`decreases` verification query.
pub(crate) fn expr_to_smt(e: &crate::ast::Expr, out: &mut String) -> bool {
    match e {
        crate::ast::Expr::Ident(s, _) => {
            out.push_str(&s.as_str());
            true
        }
        crate::ast::Expr::Literal(crate::ast::Literal::Int(v), _) => {
            out.push_str(&v.to_string());
            true
        }
        crate::ast::Expr::BinaryOp {
            op, left, right, ..
        } => {
            let smt_op = match op {
                crate::ast::BinOp::Lt => Some("<"),
                crate::ast::BinOp::Gt => Some(">"),
                crate::ast::BinOp::Le => Some("<="),
                crate::ast::BinOp::Ge => Some(">="),
                crate::ast::BinOp::Eq => Some("="),
                crate::ast::BinOp::Neq => Some("distinct"),
                crate::ast::BinOp::And => Some("and"),
                crate::ast::BinOp::Or => Some("or"),
                crate::ast::BinOp::Add => Some("+"),
                crate::ast::BinOp::Sub => Some("-"),
                crate::ast::BinOp::Mul => Some("*"),
                _ => None,
            };
            let smt_op = match smt_op {
                Some(o) => o,
                None => return false,
            };
            out.push('(');
            out.push_str(smt_op);
            out.push(' ');
            if !expr_to_smt(left, out) {
                return false;
            }
            out.push(' ');
            if !expr_to_smt(right, out) {
                return false;
            }
            out.push(')');
            true
        }
        crate::ast::Expr::UnaryOp {
            op: crate::ast::UnaryOp::Neg,
            expr,
            ..
        } => {
            out.push_str("(- 0 ");
            let ok = expr_to_smt(expr, out);
            out.push(')');
            ok
        }
        _ => false,
    }
}

/// Does the expression reference any variable in the signed set?  Used to
/// pick signed vs unsigned bit-vector comparison per operator: `Int<N>` is
/// signed (`sbvle`/`sbvge`), `UInt<N>` is unsigned (`bvule`/`bvuge`).
pub(crate) fn expr_involves_signed(
    e: &crate::ast::Expr,
    signed: &std::collections::HashSet<String>,
) -> bool {
    match e {
        crate::ast::Expr::Ident(s, _) => signed.contains(&s.as_str()),
        crate::ast::Expr::BinaryOp { left, right, .. } => {
            expr_involves_signed(left, signed) || expr_involves_signed(right, signed)
        }
        crate::ast::Expr::UnaryOp { expr, .. } => expr_involves_signed(expr, signed),
        _ => false,
    }
}

/// SMT-LIB2 bit-vector constant literal `(_ bvN W)`; negative values are
/// encoded as `(bvneg (_ bv|N| W))`.  Public counterpart of the internal
/// helper in `bii.rs` — used by the verification-side BV queries
/// (`verify_loop_decreases` bit-vector routing) to emit bound literals.
pub(crate) fn bv_const_pub(val: i128, bw: u8) -> String {
    if val < 0 {
        format!("(bvneg (_ bv{} {}))", val.unsigned_abs(), bw)
    } else {
        format!("(_ bv{} {})", val, bw)
    }
}

/// Does the expression use an explicit wrap-around operator (`+%`/`-%`/`*%`)?
/// The verification-side wrap-routing decision: an obligation or any
/// candidate hint that uses a wrap operator is only discharged under
/// BIT-VECTOR semantics (`SmtSolver::discharge_bv`) — LIA cannot express
/// modular arithmetic and must not silently accept a wrap obligation.
///
/// `callee_wraps` resolves whether a CALLED function/method is
/// wrap-semantics (its effect label carries WRAP, directly or
/// transitively) — so a wrap reached through a function call is also
/// routed to BV discharge.  The checker injects the lookup against
/// `effect_of` / `method_effect_of`; unknown callees are conservatively
/// NOT treated as wrap.  The callback receives the WHOLE callee
/// expression so the checker can distinguish a free function
/// (`Expr::Ident`) from a method call (`Expr::FieldAccess`).
pub(crate) fn expr_uses_wrap(
    e: &crate::ast::Expr,
    callee_wraps: &dyn Fn(&crate::ast::Expr) -> bool,
) -> bool {
    match e {
        crate::ast::Expr::BinaryOp {
            op, left, right, ..
        } => {
            matches!(
                op,
                crate::ast::BinOp::AddWrap
                    | crate::ast::BinOp::SubWrap
                    | crate::ast::BinOp::MulWrap
            ) || expr_uses_wrap(left, callee_wraps)
                || expr_uses_wrap(right, callee_wraps)
        }
        crate::ast::Expr::UnaryOp { expr, .. } => expr_uses_wrap(expr, callee_wraps),
        crate::ast::Expr::Call { callee, args, .. } => {
            // Wrap propagation through calls: `f(x)` / `r.foo(x)` where the
            // callee (or transitively something it calls) uses `+%`/`-%`/
            // `*%` is wrap-semantics even though no wrap operator is
            // syntactically present at this call site.
            callee_wraps(callee)
                || expr_uses_wrap(callee, callee_wraps)
                || args.iter().any(|a| expr_uses_wrap(a, callee_wraps))
        }
        _ => false,
    }
}

/// Translate a Posita invariant expression into a QF_BV / `BV` SMT-LIB2
/// term — bit-vector arithmetic at each variable's OWN bit-width from
/// `widths` (`Int<N>`/`UInt<N>` at `N` bits; a variable missing from the
/// map defaults to 64 bits).  Returns `false` for shapes the translator
/// does not support (calls, indexing, division, bit-ops, ...) — the caller
/// drops the obligation (fail-closed).  Literal constants inherit the
/// width of the sibling operand they combine with (`b <= 255` on an 8-bit
/// `b` encodes 255 at 8 bits); a literal with no variable context defaults
/// to 64 bits.  Mixing variables of DIFFERENT widths inside one expression
/// fails closed (a well-typed Posita expression never contains mixed-width
/// operands).
/// Used by the verification-side BV discharge (`SmtSolver::discharge_bv`).
/// When `signed` is `Some(set)`, comparison operators pick the SIGNED form
/// (`bvsle`/`bvsge`) for operands involving a listed variable (Posita
/// `Int<N>` — signed), and the unsigned form (`bvule`/`bvuge`) otherwise
/// (`UInt<N>` — unsigned).
pub(crate) fn expr_to_smt_bv(
    e: &crate::ast::Expr,
    out: &mut String,
    widths: &HashMap<crate::symbol::Symbol, u8>,
    signed: Option<&std::collections::HashSet<String>>,
) -> bool {
    expr_to_smt_bv_at(e, out, None, widths, signed)
}

/// The uniform bit-width of every variable in `e`: `Ok(None)` if `e` has
/// no variables, `Err(())` if it mixes variables of different widths
/// (fail closed).  A variable missing from `widths` counts as 64 bits.
fn expr_var_width(
    e: &crate::ast::Expr,
    widths: &HashMap<crate::symbol::Symbol, u8>,
) -> Result<Option<u8>, ()> {
    match e {
        crate::ast::Expr::Ident(s, _) => Ok(Some(widths.get(s).copied().unwrap_or(64))),
        crate::ast::Expr::Literal(..) => Ok(None),
        crate::ast::Expr::BinaryOp { left, right, .. } => {
            let l = expr_var_width(left, widths)?;
            let r = expr_var_width(right, widths)?;
            match (l, r) {
                (Some(a), Some(b)) if a != b => Err(()),
                _ => Ok(l.or(r)),
            }
        }
        crate::ast::Expr::UnaryOp { expr, .. } => expr_var_width(expr, widths),
        _ => Err(()),
    }
}

/// Width-aware core of `expr_to_smt_bv`.  `ctx_w` is the bit-width imposed
/// by the enclosing expression (the sibling operand's width); literals
/// translate at `ctx_w` (or 64 when no context exists).
fn expr_to_smt_bv_at(
    e: &crate::ast::Expr,
    out: &mut String,
    ctx_w: Option<u8>,
    widths: &HashMap<crate::symbol::Symbol, u8>,
    signed: Option<&std::collections::HashSet<String>>,
) -> bool {
    match e {
        crate::ast::Expr::Ident(s, _) => {
            if let Some(cw) = ctx_w
                && let Some(vw) = widths.get(s)
                && *vw != cw
            {
                return false; // mixed-width operand — fail closed.
            }
            out.push_str(&s.as_str());
            true
        }
        crate::ast::Expr::Literal(crate::ast::Literal::Int(v), _) => {
            let w = ctx_w.unwrap_or(64);
            if *v < 0 {
                out.push_str(&format!("(bvneg (_ bv{} {}))", v.unsigned_abs(), w));
            } else {
                out.push_str(&format!("(_ bv{} {})", v, w));
            }
            true
        }
        crate::ast::Expr::BinaryOp {
            op, left, right, ..
        } => {
            // Per-operand widths: mixing different widths in one
            // expression fails closed.
            let (lw, rw) = match (expr_var_width(left, widths), expr_var_width(right, widths)) {
                (Ok(l), Ok(r)) => (l, r),
                _ => return false,
            };
            if let (Some(a), Some(b)) = (lw, rw)
                && a != b
            {
                return false;
            }
            let eff = lw.or(rw);
            if let (Some(c), Some(d)) = (ctx_w, eff)
                && c != d
            {
                return false;
            }
            let w = ctx_w.or(eff).unwrap_or(64);
            // Signedness-aware comparator selection: if either operand
            // involves a signed (`Int<N>`) variable, use the signed
            // comparison; otherwise unsigned.
            let is_signed_cmp = matches!(
                op,
                crate::ast::BinOp::Lt
                    | crate::ast::BinOp::Gt
                    | crate::ast::BinOp::Le
                    | crate::ast::BinOp::Ge
            ) && signed.is_some_and(|set| {
                expr_involves_signed(left, set) || expr_involves_signed(right, set)
            });
            // Diff-row in-bounds guard: in a difference constraint
            // `x - y ≤ c` (left operand is a `Sub`), a literal bound `c`
            // must lie within the signed range (|c| < 2^(W-1)) — otherwise
            // the difference is not reliably representable in bit-vector
            // form (8-bit `255 - 0` hashes to `-1` under signed
            // interpretation) and the comparison would silently misjudge.
            // Out-of-range literal Diff bounds fail closed.  Ordinary
            // interval bounds (`x ≤ 255`) are NOT guarded — 255 is a legal
            // unsigned 8-bit maximum.
            if matches!(
                op,
                crate::ast::BinOp::Lt
                    | crate::ast::BinOp::Gt
                    | crate::ast::BinOp::Le
                    | crate::ast::BinOp::Ge
            ) && matches!(
                **left,
                crate::ast::Expr::BinaryOp {
                    op: crate::ast::BinOp::Sub,
                    ..
                }
            ) && let crate::ast::Expr::Literal(crate::ast::Literal::Int(c), _) = &**right
                && c.unsigned_abs() >= (1u128 << (w - 1))
            {
                return false;
            }
            let smt_op = match op {
                crate::ast::BinOp::Lt if is_signed_cmp => Some("bvslt"),
                crate::ast::BinOp::Gt if is_signed_cmp => Some("bvsgt"),
                crate::ast::BinOp::Le if is_signed_cmp => Some("bvsle"),
                crate::ast::BinOp::Ge if is_signed_cmp => Some("bvsge"),
                crate::ast::BinOp::Lt => Some("bvult"),
                crate::ast::BinOp::Gt => Some("bvugt"),
                crate::ast::BinOp::Le => Some("bvule"),
                crate::ast::BinOp::Ge => Some("bvuge"),
                crate::ast::BinOp::Eq => Some("="),
                crate::ast::BinOp::Neq => Some("distinct"),
                crate::ast::BinOp::And => Some("and"),
                crate::ast::BinOp::Or => Some("or"),
                crate::ast::BinOp::Add => Some("bvadd"),
                crate::ast::BinOp::Sub => Some("bvsub"),
                crate::ast::BinOp::Mul => Some("bvmul"),
                // Wrap-around operators (`+%` / `-%` / `*%`) ARE modular
                // addition/subtraction/multiplication — in bit-vector
                // semantics `bvadd`/`bvsub`/`bvmul` wrap by construction,
                // so the explicit wrap operators map directly.  The LIA
                // translator (`expr_to_smt`) deliberately does NOT accept
                // them (unbounded integers cannot express wrap-around) —
                // a wrap obligation under LIA fails closed.
                crate::ast::BinOp::AddWrap => Some("bvadd"),
                crate::ast::BinOp::SubWrap => Some("bvsub"),
                crate::ast::BinOp::MulWrap => Some("bvmul"),
                _ => None,
            };
            let smt_op = match smt_op {
                Some(o) => o,
                None => return false,
            };
            out.push('(');
            out.push_str(smt_op);
            out.push(' ');
            if !expr_to_smt_bv_at(left, out, Some(w), widths, signed) {
                return false;
            }
            out.push(' ');
            if !expr_to_smt_bv_at(right, out, Some(w), widths, signed) {
                return false;
            }
            out.push(')');
            true
        }
        crate::ast::Expr::UnaryOp {
            op: crate::ast::UnaryOp::Neg,
            expr,
            ..
        } => {
            out.push_str("(bvneg ");
            let ok = expr_to_smt_bv_at(expr, out, ctx_w, widths, signed);
            out.push(')');
            ok
        }
        _ => false,
    }
}

/// Collect the identifiers occurring in an expression (as SMT variable
/// names) — the `declare-const` set for a hint query.
fn collect_idents(e: &crate::ast::Expr, out: &mut Vec<String>) {
    match e {
        crate::ast::Expr::Ident(s, _) => out.push(s.as_str()),
        crate::ast::Expr::Literal(..) => {}
        crate::ast::Expr::BinaryOp { left, right, .. } => {
            collect_idents(left, out);
            collect_idents(right, out);
        }
        crate::ast::Expr::UnaryOp { expr, .. } => collect_idents(expr, out),
        _ => {}
    }
}

impl SmtSolver {
    /// The `@hint(assertion)` injection gate — check whether invariant
    /// candidate assertions are CONSISTENT (Z3 sat).  `true` means the
    /// candidates can be seeded into the solver context as hints; `false`
    /// (unsat / timeout / solver error / untranslatable expression) means
    /// the candidates are dropped.  Per the 2026-08-13 committee ruling,
    /// this check NEVER discharges obligations by itself — the solver
    /// remains the authority.
    pub fn check_hints<'input>(&self, assertions: &[&crate::ast::Expr<'input>]) -> bool {
        let mut smt = String::new();
        smt.push_str("(set-logic LIA)\n");
        let mut vars: Vec<String> = Vec::new();
        for a in assertions {
            collect_idents(a, &mut vars);
        }
        vars.sort_unstable();
        vars.dedup();
        for v in &vars {
            smt.push_str(&format!("(declare-const {} Int)\n", v));
        }
        for a in assertions {
            let mut e = String::new();
            if !expr_to_smt(a, &mut e) {
                return false; // untranslatable → drop the candidate.
            }
            smt.push_str(&format!("(assert {})\n", e));
        }
        smt.push_str("(check-sat)\n");
        // `call_z3` reports success by exit code and wraps the RAW stdout
        // (the unicity query parses it line-by-line); here the last
        // sat/unsat result line decides.  Anything else (unsat, timeout,
        // error, unparsable) drops the candidate — fail-closed.
        match self.call_z3(&smt) {
            SmtResult::Sat(out) => out
                .lines()
                .rev()
                .find_map(|line| match line.trim() {
                    "sat" => Some(true),
                    "unsat" => Some(false),
                    _ => None,
                })
                .unwrap_or(false),
            _ => false,
        }
    }

    /// Run a caller-built SMT-LIB2 query (logic, declarations, asserts,
    /// `check-sat`, optional `get-model`) and classify the outcome with an
    /// explicit tri-state.  Used by the BII synthesis `Refine` step for its
    /// ∃∀ induction queries, where `unknown` (undecided) must be
    /// distinguished from `sat`/`unsat` — the caller fails closed on
    /// `Unknown`.
    pub fn run_raw_query(&self, smt: &str) -> RawQueryOutcome {
        let raw = match self.call_z3(smt) {
            SmtResult::Sat(out) => out,
            SmtResult::Unsat => return RawQueryOutcome::Unsat,
            SmtResult::Unknown | SmtResult::Timeout => return RawQueryOutcome::Unknown,
            SmtResult::Error(e) => return RawQueryOutcome::Error(e),
        };
        // `call_z3` classifies by EXIT CODE, so a successful `unsat`/`sat`
        // output arrives inside `SmtResult::Sat(raw)`.  The decisive
        // result is the LAST sat/unsat/unknown line of the raw output —
        // classify from the text, not the process status.
        match raw.lines().rev().find_map(|line| match line.trim() {
            "sat" => Some(RawQueryOutcome::Sat(raw.clone())),
            "unsat" => Some(RawQueryOutcome::Unsat),
            "unknown" => Some(RawQueryOutcome::Unknown),
            _ => None,
        }) {
            Some(o) => o,
            None => RawQueryOutcome::Error("no sat/unsat/unknown line in output".into()),
        }
    }

    /// Discharge an obligation from candidate hints —
    /// `∧hints ⟹ obligation`.  The query asserts every hint plus the
    /// NEGATION of the obligation; `unsat` proves the entailment
    /// (discharge succeeds).  `sat` (a counter-model exists) and every
    /// fail-closed outcome (`unknown`, timeout, solver error, or an
    /// untranslatable expression) return `false` — the caller keeps the
    /// conservative path (hint-only / reject).
    pub fn discharge<'input>(
        &self,
        hints: &[&crate::ast::Expr<'input>],
        obligation: &crate::ast::Expr<'input>,
    ) -> bool {
        let mut smt = String::new();
        smt.push_str("(set-logic LIA)\n");
        let mut vars: Vec<String> = Vec::new();
        for h in hints {
            collect_idents(h, &mut vars);
        }
        collect_idents(obligation, &mut vars);
        vars.sort_unstable();
        vars.dedup();
        for v in &vars {
            smt.push_str(&format!("(declare-const {} Int)\n", v));
        }
        for h in hints {
            let mut e = String::new();
            if !expr_to_smt(h, &mut e) {
                return false; // untranslatable hint — fail closed.
            }
            smt.push_str(&format!("(assert {})\n", e));
        }
        // The negation of the obligation: if satisfiable together with the
        // hints, a counter-model exists and the obligation is NOT entailed.
        let mut neg = String::new();
        if !expr_to_smt(obligation, &mut neg) {
            return false; // untranslatable obligation — fail closed.
        }
        smt.push_str(&format!("(assert (not {}))\n", neg));
        smt.push_str("(check-sat)\n");
        match self.run_raw_query(&smt) {
            RawQueryOutcome::Unsat => true, // ∧hints ∧ ¬obligation unsat → entailed.
            RawQueryOutcome::Sat(_) | RawQueryOutcome::Unknown | RawQueryOutcome::Error(_) => false,
        }
    }

    /// Verification-side discharge: an obligation from candidate hints
    /// under BIT-VECTOR semantics — `∧hints ⟹ obligation` with the paper's
    /// modular (wrap-around) arithmetic.  The query declares every variable
    /// at its OWN width from `widths` (`Int<N>`/`UInt<N>` at `N` bits; a
    /// variable missing from the map defaults to 64 bits) and translates
    /// hints and the negated obligation via `expr_to_smt_bv`
    /// (`bvadd`/`bvsub`/`bvule`…).
    ///
    /// This is the verification counterpart to the synthesis-side
    /// `use_bv: true` encoding: a candidate synthesized under wrap-around
    /// semantics (e.g. an 8-bit counter's BII with `ub = 255`) can only be
    /// accepted here — under LIA, `255+1 = 256` would falsify it.  Any
    /// untranslatable hint/obligation fails closed (returns `false`).
    ///
    /// Signedness refinement (`Int<N>` → `bvsle`, `UInt<N>` → `bvule`) is
    /// enabled by passing `Some(signed)` — the set of variable names that
    /// are SIGNED (`Int<N>`); pass `None` for uniform unsigned comparison.
    /// Discharge `∧hints ⟹ obligation` under bit-vector (wrap-around)
    /// semantics: assert the negation of the obligation, expect `unsat`.
    /// `widths` gives each variable's bit-width (unknown → 64); `signed`
    /// selects signed (`bvsle`/`bvslt`) vs unsigned (`bvule`/`bvult`)
    /// comparators for the HINTS and the obligation.
    ///
    /// `hints_unsigned` overrides the HINT encoding only: when the hints
    /// are BII-synthesized template rows (proposed and validated UNSIGNED
    /// over the template domain), re-encoding them at the guard's
    /// signedness can empty a sign-boundary-crossing row (e.g. [127, 129]
    /// on 8-bit) and make the premise — and therefore the proof —
    /// vacuously true.  Pass `true` to keep such rows intact; the
    /// obligation keeps the `signed` reading.  DBM-origin hints with
    /// negative bounds (`i >= -5`) must pass `false` — forced-unsigned
    /// encoding would misread the two's-complement pattern.
    pub fn discharge_bv<'input>(
        &self,
        hints: &[&crate::ast::Expr<'input>],
        obligation: &crate::ast::Expr<'input>,
        widths: &HashMap<crate::symbol::Symbol, u8>,
        signed: Option<&std::collections::HashSet<String>>,
        hints_unsigned: bool,
    ) -> bool {
        let mut smt = String::new();
        smt.push_str("(set-logic BV)\n");
        let mut vars: Vec<String> = Vec::new();
        for h in hints {
            collect_idents(h, &mut vars);
        }
        collect_idents(obligation, &mut vars);
        vars.sort_unstable();
        vars.dedup();
        for v in &vars {
            let w = widths
                .get(&crate::symbol::Symbol::intern(v))
                .copied()
                .unwrap_or(64);
            smt.push_str(&format!("(declare-const {} (_ BitVec {}))\n", v, w));
        }
        for h in hints {
            let mut e = String::new();
            let hint_signed = if hints_unsigned { None } else { signed };
            if !expr_to_smt_bv(h, &mut e, widths, hint_signed) {
                return false; // untranslatable hint — fail closed.
            }
            smt.push_str(&format!("(assert {})\n", e));
        }
        // The negation of the obligation: satisfiable with the hints ⇒ a
        // counter-model exists under wrap-around semantics ⇒ NOT entailed.
        let mut neg = String::new();
        if !expr_to_smt_bv(obligation, &mut neg, widths, signed) {
            return false; // untranslatable obligation — fail closed.
        }
        smt.push_str(&format!("(assert (not {}))\n", neg));
        smt.push_str("(check-sat)\n");
        match self.run_raw_query(&smt) {
            RawQueryOutcome::Unsat => true, // ∧hints ∧ ¬obligation unsat → entailed.
            RawQueryOutcome::Sat(_) | RawQueryOutcome::Unknown | RawQueryOutcome::Error(_) => false,
        }
    }

    fn call_z3(&self, smt: &str) -> SmtResult {
        if smt.is_empty() {
            return SmtResult::Error("empty query".into());
        }

        // Check the query cache first — avoids spawning Z3 for identical queries.
        {
            let cache = self.query_cache.borrow();
            if let Some(cached) = cache.get(smt) {
                return cached.clone();
            }
        }

        // Build the SMT query with timeout and memory limit baked in.
        let mut smt_with_limits = String::new();
        smt_with_limits.push_str(&format!(
            "(set-option :timeout {})\n",
            *self.timeout_ms.borrow()
        ));
        smt_with_limits.push_str(&format!(
            "(set-option :memory_max_size {})\n",
            Z3_MEMORY_LIMIT_MB
        ));
        smt_with_limits.push_str(smt);

        let mut child = match Command::new(&self.solver_path)
            .arg("-in")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(c) => KillOnDropChild(c),
            Err(e) => {
                Z3_WARNED.get_or_init(|| {
                    eprintln!("warning: SMT solver ({}) not found: {}; unicity check uses fallback heuristic", self.solver_path, e);
                    true
                });
                return SmtResult::Error(format!("solver not found: {}", e));
            }
        };

        if let Some(mut stdin) = child.0.stdin.take()
            && stdin.write_all(smt_with_limits.as_bytes()).is_err()
        {
            // The guard's Drop kills the child on this return path.
            return SmtResult::Error("stdin write failed".into());
        }

        let output = match child.wait_with_output() {
            Ok(o) => o,
            // The guard's Drop kills the child on this return path too.
            Err(e) => return SmtResult::Error(format!("wait failed: {}", e)),
        };

        let result = if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            // Exit code 0 does NOT imply sat: z3 prints `unknown`
            // (undecided) with exit code 0.  Treat any undecided result
            // line as `Unknown` so callers fail closed instead of
            // misreading it as sat (a latent soundness hazard for
            // quantified queries).
            if stdout.lines().any(|l| l.trim() == "unknown") {
                SmtResult::Unknown
            } else {
                SmtResult::Sat(stdout)
            }
        } else {
            // Check stderr for timeout indicator.
            let stderr = String::from_utf8_lossy(&output.stderr);
            if stderr.contains("timeout") {
                SmtResult::Timeout
            } else {
                SmtResult::Error(format!("z3 error: {}", stderr.trim()))
            }
        };

        // Cache the result before returning.
        self.query_cache
            .borrow_mut()
            .insert(smt.to_string(), result.clone());
        result
    }

    fn parse_unicity_results(
        output: &str,
        shape_names: &[(&str, PrincipalShape)],
    ) -> Option<PrincipalShape> {
        let mut unique_shape: Option<PrincipalShape> = None;
        // Collect all sat/unsat results in order, skipping non-result lines.
        let results: Vec<bool> = output
            .lines()
            .filter_map(|line| match line.trim() {
                "sat" => Some(true),
                "unsat" => Some(false),
                _ => None,
            })
            .collect();
        for (i, &is_sat) in results.iter().enumerate() {
            if is_sat && i < shape_names.len() {
                if unique_shape.is_some() {
                    return None; // multiple shapes possible — ambiguous
                }
                unique_shape = Some(shape_names[i].1);
            }
        }
        unique_shape
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{BinOp, Expr, Literal, Span};
    use crate::symbol::Symbol;

    fn ident<'a>(arena: &'a bumpalo::Bump, name: &str) -> &'a Expr<'a> {
        arena.alloc(Expr::Ident(Symbol::intern(name), Span::new(0, 0)))
    }

    fn lit<'a>(arena: &'a bumpalo::Bump, v: i128) -> &'a Expr<'a> {
        arena.alloc(Expr::Literal(
            Literal::Int(crate::ast::IntLit::Small(v)),
            Span::new(0, 0),
        ))
    }

    fn bin<'a>(
        arena: &'a bumpalo::Bump,
        op: BinOp,
        l: &'a Expr<'a>,
        r: &'a Expr<'a>,
    ) -> &'a Expr<'a> {
        arena.alloc(Expr::BinaryOp {
            left: l,
            op,
            right: r,
            span: Span::new(0, 0),
        })
    }

    /// The expression translator produces SMT-LIB2 terms.
    #[test]
    fn test_expr_to_smt() {
        let arena: &'static bumpalo::Bump = Box::leak(Box::new(bumpalo::Bump::new()));
        let e = bin(&arena, BinOp::Ge, ident(&arena, "i"), lit(&arena, 0));
        let mut s = String::new();
        assert!(expr_to_smt(e, &mut s));
        assert_eq!(s, "(>= i 0)");
        let e2 = bin(
            &arena,
            BinOp::And,
            e,
            bin(&arena, BinOp::Lt, ident(&arena, "i"), lit(&arena, 10)),
        );
        let mut s2 = String::new();
        assert!(expr_to_smt(e2, &mut s2));
        assert_eq!(s2, "(and (>= i 0) (< i 10))");
    }

    /// The hint gate — a consistent candidate is seeded, a
    /// contradictory one is dropped.  Skipped when Z3 is unavailable.
    #[test]
    fn test_check_hints() {
        let solver = SmtSolver::new("z3");
        if !solver.check_version() {
            eprintln!("z3 unavailable — skipping test_check_hints");
            return;
        }
        let arena: &'static bumpalo::Bump = Box::leak(Box::new(bumpalo::Bump::new()));
        let ge0 = bin(&arena, BinOp::Ge, ident(&arena, "i"), lit(&arena, 0));
        assert!(solver.check_hints(&[ge0]), "i ≥ 0 is consistent");
        let contradictory = bin(
            &arena,
            BinOp::And,
            ge0,
            bin(&arena, BinOp::Le, ident(&arena, "i"), lit(&arena, -1)),
        );
        assert!(
            !solver.check_hints(&[contradictory]),
            "i ≥ 0 ∧ i ≤ -1 is inconsistent — dropped"
        );
    }

    /// Wrap-routing detection: `expr_uses_wrap` detects explicit
    /// wrap-around operators (`+%`/`-%`/`*%`) and nothing else — the
    /// decision that sends an obligation to bit-vector discharge.
    #[test]
    fn test_expr_uses_wrap() {
        let arena: &'static bumpalo::Bump = Box::leak(Box::new(bumpalo::Bump::new()));
        let x = ident(&arena, "x");
        let one = lit(&arena, 1);
        let wrap = bin(&arena, BinOp::AddWrap, x, one);
        assert!(
            expr_uses_wrap(wrap, &|_| false),
            "`x +% 1` uses a wrap operator"
        );
        let plain = bin(&arena, BinOp::Add, x, one);
        assert!(!expr_uses_wrap(plain, &|_| false), "`x + 1` does not wrap");
        let nested = bin(&arena, BinOp::Ge, wrap, lit(&arena, 0));
        assert!(
            expr_uses_wrap(nested, &|_| false),
            "wrap detection must recurse into sub-expressions"
        );
        // A CALL to a wrap-semantics function propagates WRAP through the
        // call graph even though no wrap operator is syntactically present
        // at the call site (`y = f(x)` where `f` uses `x +% 1`).
        let f = ident(&arena, "f");
        let call = arena.alloc(Expr::Call {
            callee: f,
            args: vec![Expr::Ident(Symbol::intern("x"), Span::new(0, 0))],
            comptime: false,
            span: Span::new(0, 0),
        });
        assert!(
            expr_uses_wrap(call, &|callee| matches!(
                callee,
                crate::ast::Expr::Ident(f, _) if f.eq_str("f")
            )),
            "a call to a wrap-semantics function must be recognized as wrap"
        );
        // A call to a NON-wrap function is not wrap.
        let call_plain = arena.alloc(Expr::Call {
            callee: f,
            args: vec![Expr::Ident(Symbol::intern("x"), Span::new(0, 0))],
            comptime: false,
            span: Span::new(0, 0),
        });
        assert!(
            !expr_uses_wrap(call_plain, &|_| false),
            "a call to a non-wrap function is not wrap"
        );
        // A METHOD call `r.foo()` whose receiver's method is wrap-semantics
        // must also be recognized: the callback receives the whole callee
        // (`Expr::FieldAccess`), and the checker-side routing resolves it
        // through `method_effect_of`.
        let r = ident(&arena, "r");
        let field = arena.alloc(Expr::FieldAccess {
            base: r,
            field: Symbol::intern("foo"),
            span: Span::new(0, 0),
        });
        let method_call = arena.alloc(Expr::Call {
            callee: field,
            args: vec![Expr::Ident(Symbol::intern("x"), Span::new(0, 0))],
            comptime: false,
            span: Span::new(0, 0),
        });
        assert!(
            expr_uses_wrap(method_call, &|callee| matches!(
                callee,
                crate::ast::Expr::FieldAccess { field, .. } if field.eq_str("foo")
            )),
            "a method call to a wrap-semantics method must be recognized as wrap"
        );
        // A method call whose method does NOT wrap is not wrap.
        assert!(
            !expr_uses_wrap(method_call, &|_| false),
            "a method call to a non-wrap method is not wrap"
        );
    }

    /// Wrap-operator translation: `expr_to_smt_bv` maps wrap operators to
    /// `bvadd`/`bvsub`/`bvmul` and emits bit-vector literals (incl.
    /// negatives via `bvneg`) at the variable's own width.
    #[test]
    fn test_expr_to_smt_bv_wrap_and_literals() {
        let arena: &'static bumpalo::Bump = Box::leak(Box::new(bumpalo::Bump::new()));
        let x = ident(&arena, "x");
        let wrap = bin(&arena, BinOp::AddWrap, x, lit(&arena, 1));
        let mut s = String::new();
        assert!(expr_to_smt_bv(
            wrap,
            &mut s,
            &HashMap::from([(Symbol::intern("x"), 8u8)]),
            None
        ));
        assert_eq!(s, "(bvadd x (_ bv1 8))");
        // Negative literal with no variable context defaults to 64 bits.
        let mut neg = String::new();
        assert!(expr_to_smt_bv(
            lit(&arena, -5),
            &mut neg,
            &HashMap::new(),
            None
        ));
        assert_eq!(neg, "(bvneg (_ bv5 64))");
        // The literal inherits the sibling variable's width (`x - 5` on an
        // 8-bit `x` encodes 5 at 8 bits).
        let mut sub = String::new();
        let sub_expr = bin(&arena, BinOp::Sub, ident(&arena, "x"), lit(&arena, 5));
        assert!(expr_to_smt_bv(
            sub_expr,
            &mut sub,
            &HashMap::from([(Symbol::intern("x"), 8u8)]),
            None
        ));
        assert_eq!(sub, "(bvsub x (_ bv5 8))");
    }

    /// Diff-row in-bounds guard: a literal bound `c` in `x - y ≤ c`
    /// with |c| ≥ 2^(W-1) fails closed (cannot be represented reliably),
    /// while an in-range bound translates normally.
    #[test]
    fn test_expr_to_smt_bv_diff_in_bounds_guard() {
        let arena: &'static bumpalo::Bump = Box::leak(Box::new(bumpalo::Bump::new()));
        let x = ident(&arena, "x");
        let y = ident(&arena, "y");
        // |128| = 2^(8-1): at/above the boundary → out of range → fail.
        let out = bin(
            &arena,
            BinOp::Le,
            bin(&arena, BinOp::Sub, x, y),
            lit(&arena, 128),
        );
        let mut s = String::new();
        assert!(
            !expr_to_smt_bv(
                out,
                &mut s,
                &HashMap::from([(Symbol::intern("x"), 8u8), (Symbol::intern("y"), 8u8)]),
                None
            ),
            "literal Diff bound |c| ≥ 2^(W-1) must fail closed"
        );
        // |127| < 128: in range (signed 8-bit max) → translates.
        let inside = bin(
            &arena,
            BinOp::Le,
            bin(&arena, BinOp::Sub, x, y),
            lit(&arena, 127),
        );
        let mut s2 = String::new();
        assert!(expr_to_smt_bv(
            inside,
            &mut s2,
            &HashMap::from([(Symbol::intern("x"), 8u8), (Symbol::intern("y"), 8u8)]),
            None
        ));
        assert!(s2.contains("bvsub"), "in-range Diff bound uses bvsub");
    }

    /// Bit-vector discharge: `discharge_bv` accepts a wrap-around
    /// obligation that LIA would reject — the 8-bit counter `x := 255;
    /// while x ≤ 255 { x := x +% 1 }` has BII `x ≤ 255` (255+1 wraps to 0).
    #[test]
    fn test_discharge_bv_wrap_obligation_accepted() {
        let solver = SmtSolver::new("z3");
        if !solver.check_version() {
            eprintln!("z3 unavailable — skipping test_discharge_bv_wrap_obligation_accepted");
            return;
        }
        let arena: &'static bumpalo::Bump = Box::leak(Box::new(bumpalo::Bump::new()));
        // hints: x ≥ 0 ∧ x ≤ 255 (the invariant candidates)
        let x = ident(&arena, "x");
        let ge0 = bin(&arena, BinOp::Ge, x, lit(&arena, 0));
        let le255 = bin(&arena, BinOp::Le, x, lit(&arena, 255));
        let hints = [ge0, le255];
        // obligation: x +% 1 ≤ 255 — under 8-bit wrap-around semantics this
        // is entailed by the hints (255+1 wraps to 0); under LIA it is not.
        let obligation = bin(
            &arena,
            BinOp::Le,
            bin(&arena, BinOp::AddWrap, x, lit(&arena, 1)),
            lit(&arena, 255),
        );
        assert!(
            solver.discharge_bv(
                &hints,
                obligation,
                &HashMap::from([(Symbol::intern("x"), 8u8)]),
                None,
                false
            ),
            "wrap-around obligation must be entailed under BV semantics"
        );
    }

    /// Signedness refinement: with a signed variable set, the translator
    /// emits the STANDARD signed bit-vector comparators `bvsle`/`bvsge`
    /// (not the non-standard `sbvle`/`sbvge` — Z3 rejects those as
    /// "unknown constant", verified empirically).  The emitted query must
    /// also be accepted by Z3.
    #[test]
    fn test_expr_to_smt_bv_signed_comparator_standard_name() {
        let arena: &'static bumpalo::Bump = Box::leak(Box::new(bumpalo::Bump::new()));
        let x = ident(&arena, "x");
        let le = bin(&arena, BinOp::Le, x, lit(&arena, 127));
        let mut s = String::new();
        let mut signed = std::collections::HashSet::new();
        signed.insert("x".to_string());
        assert!(expr_to_smt_bv(
            le,
            &mut s,
            &HashMap::from([(Symbol::intern("x"), 8u8)]),
            Some(&signed)
        ));
        assert!(
            s.contains("bvsle"),
            "signed comparison must use the standard `bvsle` (got: {})",
            s
        );
        assert!(
            !s.contains("sbvle"),
            "non-standard `sbvle` must not be emitted (got: {})",
            s
        );
        // The emitted term must be accepted by Z3 (unknown comparator =
        // rejected).
        let solver = SmtSolver::new("z3");
        if !solver.check_version() {
            eprintln!("z3 unavailable — skipping Z3-acceptance half of the test");
            return;
        }
        let query = format!(
            "(set-logic BV)\n(declare-const x (_ BitVec 8))\n(assert {})\n(check-sat)\n",
            s
        );
        match solver.run_raw_query(&query) {
            crate::hir::smt::RawQueryOutcome::Sat(_) | crate::hir::smt::RawQueryOutcome::Unsat => {}
            other => panic!("Z3 must accept the bvsle query, got {:?}", other),
        }
    }

    /// Per-variable signedness production wiring: a NEGATIVE-bound
    /// obligation `x ≥ -5` on `Int<8>` (two's-complement `0xFB`) must be
    /// compared with the SIGNED comparator `bvsge` when `x` is in the
    /// signed set — under unsigned `bvuge`, `0xFB` (251) would pass `≥ 5`
    /// checks that should fail (or vice versa for `x ≤ -5`).
    #[test]
    fn test_expr_to_smt_bv_negative_bound_signed_vs_unsigned() {
        let arena: &'static bumpalo::Bump = Box::leak(Box::new(bumpalo::Bump::new()));
        let x = ident(&arena, "x");
        // x ≥ -5  →  in signed 8-bit terms: x ∈ [-5, 127].
        let ge = bin(&arena, BinOp::Ge, x, lit(&arena, -5));
        let mut signed_out = String::new();
        let mut signed = std::collections::HashSet::new();
        signed.insert("x".to_string());
        assert!(expr_to_smt_bv(
            ge,
            &mut signed_out,
            &HashMap::from([(Symbol::intern("x"), 8u8)]),
            Some(&signed)
        ));
        assert!(
            signed_out.contains("bvsge"),
            "Int<N> variable must use the signed comparator bvsge (got: {})",
            signed_out
        );
        assert!(
            !signed_out.contains("bvuge"),
            "signed variable must NOT use unsigned bvuge (got: {})",
            signed_out
        );
        assert!(
            signed_out.contains("(bvneg (_ bv5 8))"),
            "negative literal -5 must be encoded as bvneg (got: {})",
            signed_out
        );
        // Same expression WITHOUT the signed set → unsigned comparison.
        let mut unsigned_out = String::new();
        assert!(expr_to_smt_bv(
            ge,
            &mut unsigned_out,
            &HashMap::from([(Symbol::intern("x"), 8u8)]),
            None
        ));
        assert!(
            unsigned_out.contains("bvuge"),
            "no signed set → unsigned bvuge (got: {})",
            unsigned_out
        );
    }
}
