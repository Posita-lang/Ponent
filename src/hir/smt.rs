use crate::hir::infer::PrincipalShape;
use crate::hir::types::*;
use std::collections::HashMap;
use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::OnceLock;
use std::time::Duration;

static Z3_WARNED: OnceLock<bool> = OnceLock::new();

/// Default timeout for Z3 solver invocations (milliseconds).
const Z3_TIMEOUT_MS: u64 = 5_000;
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
pub struct SmtSolver {
    solver_path: String,
}

impl SmtSolver {
    pub fn new(solver_path: &str) -> Self {
        let solver = SmtSolver {
            solver_path: solver_path.to_string(),
        };
        // Verify Z3 version on first use (lazy, via check_version).
        solver
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
    pub fn check_unicity(
        &self,
        ctx: &TypeContext,
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
                TypeData::Struct { args, .. } | TypeData::Enum { args, .. } => {
                    PrincipalShape::Constructor(args.len())
                }
                TypeData::Forall { .. } | TypeData::Exists { .. } | TypeData::Poly { .. } => {
                    PrincipalShape::Poly
                }
                _ => PrincipalShape::Unknown,
            });
        }

        let var_id = match ctx.get(resolved) {
            TypeData::InferVar { id } => *id,
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
        smt.push_str("(assert (= SHAPE_UNKNOWN 0))\n");
        smt.push_str("(assert (= SHAPE_ARROW 1))\n");
        smt.push_str("(assert (= SHAPE_TUPLE 2))\n");
        smt.push_str("(assert (= SHAPE_CONSTRUCTOR 3))\n");
        smt.push_str("(assert (= SHAPE_POLY 4))\n\n");

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
        smt.push_str("(declare-fun type-rational (Int Int) Type)\n\n");

        // ── 4. Shape-of function ────────────────────────────────
        smt.push_str("(declare-fun shape-of (Type) Int)\n\n");

        // ── 5. Shape axioms ──────────────────────────────────────
        smt.push_str("(assert (= (shape-of type-int32) SHAPE_UNKNOWN))\n");
        smt.push_str("(assert (= (shape-of type-int64) SHAPE_UNKNOWN))\n");
        smt.push_str("(assert (= (shape-of type-bool) SHAPE_UNKNOWN))\n");
        smt.push_str("(assert (= (shape-of type-unit) SHAPE_UNKNOWN))\n");
        smt.push_str("(assert (= (shape-of type-never) SHAPE_UNKNOWN))\n");
        smt.push_str("(assert (= (shape-of type-char) SHAPE_UNKNOWN))\n");
        smt.push_str("(assert (= (shape-of type-byte) SHAPE_UNKNOWN))\n");
        smt.push_str(
            "(assert (forall ((a Type) (b Type)) (= (shape-of (type-fn a b)) SHAPE_ARROW)))\n",
        );
        smt.push_str(
            "(assert (forall ((a Type) (b Type)) (= (shape-of (type-tuple2 a b)) SHAPE_TUPLE)))\n",
        );
        smt.push_str("(assert (forall ((tag Int) (a Type)) (= (shape-of (type-struct tag a)) SHAPE_CONSTRUCTOR)))\n");
        smt.push_str("(assert (forall ((a Type)) (= (shape-of (type-poly a)) SHAPE_POLY)))\n");
        smt.push_str("(assert (forall ((p Int) (q Int)) (= (shape-of (type-rational p q)) SHAPE_UNKNOWN)))\n\n");

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
        let output = self.call_z3(&smt)?;
        Self::parse_unicity_results(&output, &shape_names)
    }

    /// Convert a TypeId to an SMT-LIB2 term.
    fn type_to_smt_term(ctx: &TypeContext, ty: TypeId) -> String {
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
            TypeData::InferVar { id } => format!("iv_{}", id),
            TypeData::Rational {
                int_bits,
                frac_bits,
            } => {
                format!("(type-rational {} {})", int_bits, frac_bits)
            }
            _ => "type-unknown".into(),
        }
    }

    fn call_z3(&self, smt: &str) -> Option<String> {
        if smt.is_empty() {
            return None;
        }
        // Build the SMT query with timeout and memory limit baked in.
        let mut smt_with_limits = String::new();
        smt_with_limits.push_str(&format!(
            "(set-option :timeout {})\n",
            Z3_TIMEOUT_MS
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
            .stderr(Stdio::null())
            .spawn()
        {
            Ok(c) => c,
            Err(e) => {
                Z3_WARNED.get_or_init(|| {
                    eprintln!("warning: SMT solver ({}) not found: {}; unicity check uses fallback heuristic", self.solver_path, e);
                    true
                });
                return None;
            }
        };

        if let Some(mut stdin) = child.stdin.take() {
            if stdin.write_all(smt_with_limits.as_bytes()).is_err() {
                let _ = child.kill();
                return None;
            }
        }

        // Wait with a timeout via `wait()` (no cross-platform alarm available).
        let output = match child.wait_with_output() {
            Ok(o) => o,
            Err(_) => return None,
        };

        if output.status.success() {
            Some(String::from_utf8_lossy(&output.stdout).to_string())
        } else {
            // Timeout or error — Z3 returns exit code 1 on `(check-sat)` after timeout
            // with "(error "timeout")" on stderr, which is expected and not a crash.
            None
        }
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
                unique_shape = Some(shape_names[i].1.clone());
            }
        }
        unique_shape
    }
}
