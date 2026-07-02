use crate::hir::infer::PrincipalShape;
use crate::hir::types::*;
use std::collections::HashMap;
use std::io::Write;
use std::process::{Command, Stdio};

/// SMT-LIB2-based unicity checker using Z3.
///
/// Encodes the constraint context `C` as first-order formulas over an
/// uninterpreted sort `Type`, then queries Z3 for the unique shape of a
/// target variable (O'Brien, Rémy & Scherer §4.1):
///
///   C[τ!ζ] iff ∀φ, φ ⊢ [C[τ = g]] ⇒ shape(g) = ζ
///
/// Algorithm:
/// 1. Declare `Type` as an uninterpreted sort
/// 2. Declare constructor functions (int32, bool, fn, tuple, etc.)
/// 3. Declone each InferVar as a constant of sort `Type`
/// 4. Assert all known bindings and Eq constraints as equalities
/// 5. Define `shape-of : Type → Int` returning 0..N for each shape class
/// 6. For each candidate shape tag s, push/assert (= (shape-of τ) s)/check-sat/pop
/// 7. If exactly one candidate yields sat, that shape is unique
pub struct SmtSolver {
    z3_path: String,
}

impl SmtSolver {
    pub fn new(z3_path: &str) -> Self {
        SmtSolver {
            z3_path: z3_path.to_string(),
        }
    }

    /// Main entry: check whether `ty` (an InferVar or resolved type) has a
    /// unique shape given the constraint context.
    ///
    /// `solver_ctx` should contain the SMT formulas for all active constraints.
    /// If `ty` is already concrete, the shape is trivially known.
    pub fn check_unicity(
        &self,
        ctx: &TypeContext,
        ty: TypeId,
        solver_ctx: &str,
    ) -> Option<PrincipalShape> {
        let resolved = ctx.resolve_binding(ty);
        // If already resolved to a concrete type, shape is known immediately.
        if !matches!(ctx.get(resolved), TypeData::InferVar { .. }) {
            let shape = match ctx.get(resolved) {
                TypeData::Fn { .. } => PrincipalShape::Arrow,
                TypeData::Tuple { elems } => PrincipalShape::Tuple(elems.len()),
                TypeData::Struct { args, .. } | TypeData::Enum { args, .. } => {
                    PrincipalShape::Constructor(args.len())
                }
                TypeData::Forall { .. } | TypeData::Exists { .. } | TypeData::Poly { .. } => PrincipalShape::Poly,
                _ => PrincipalShape::Unknown,
            };
            return Some(shape);
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

        // ── 2. Declare type constructors as uninterpreted functions ──
        smt.push_str("; Type constructor tags (returned by shape-of)\n");
        smt.push_str("(declare-const SHAPE_UNKNOWN Int)\n");
        smt.push_str("(declare-const SHAPE_ARROW Int)\n");
        smt.push_str("(declare-const SHAPE_TUPLE Int)\n");
        smt.push_str("(declare-const SHAPE_CONSTRUCTOR Int)\n");
        smt.push_str("(declare-const SHAPE_POLY Int)\n");
        smt.push_str("(declare-const SHAPE_VAR Int)\n");
        smt.push_str("(assert (= SHAPE_UNKNOWN 0))\n");
        smt.push_str("(assert (= SHAPE_ARROW 1))\n");
        smt.push_str("(assert (= SHAPE_TUPLE 2))\n");
        smt.push_str("(assert (= SHAPE_CONSTRUCTOR 3))\n");
        smt.push_str("(assert (= SHAPE_POLY 4))\n");
        smt.push_str("(assert (= SHAPE_VAR 5))\n\n");

        // ── 3. Declare uninterpreted functions for type constructors ──
        smt.push_str("(declare-fun type-int32 () Type)\n");
        smt.push_str("(declare-fun type-int64 () Type)\n");
        smt.push_str("(declare-fun type-bool () Type)\n");
        smt.push_str("(declare-fun type-unit () Type)\n");
        smt.push_str("(declare-fun type-never () Type)\n");
        smt.push_str("(declare-fun type-char () Type)\n");
        smt.push_str("(declare-fun type-byte () Type)\n");
        // Fn: (Type × Type) → Type  (param type → return type → Type)
        smt.push_str("(declare-fun type-fn (Type Type) Type)\n");
        // Tuple: for simplicity, only encode Tuple2 (pair)
        smt.push_str("(declare-fun type-tuple2 (Type Type) Type)\n");
        // Named constructors: (Int → Type) — using a simple tag
        smt.push_str("(declare-fun type-struct (Int Type) Type)\n");
        // Polymorphic container
        smt.push_str("(declare-fun type-poly (Type) Type)\n\n");

        // ── 4. Declare shape-of function ─────────────────────────
        // shape-of : Type → Int
        smt.push_str("(declare-fun shape-of (Type) Int)\n\n");

        // ── 5. Axiomatise shape-of for each constructor ────────────
        smt.push_str("; Axioms: shape-of for each type constructor\n");
        smt.push_str("(assert (= (shape-of type-int32) SHAPE_UNKNOWN))\n");
        smt.push_str("(assert (= (shape-of type-int64) SHAPE_UNKNOWN))\n");
        smt.push_str("(assert (= (shape-of type-bool) SHAPE_UNKNOWN))\n");
        smt.push_str("(assert (= (shape-of type-unit) SHAPE_UNKNOWN))\n");
        smt.push_str("(assert (= (shape-of type-never) SHAPE_UNKNOWN))\n");
        smt.push_str("(assert (= (shape-of type-char) SHAPE_UNKNOWN))\n");
        smt.push_str("(assert (= (shape-of type-byte) SHAPE_UNKNOWN))\n");
        // Fn has shape ARROW
        smt.push_str("(assert (forall ((a Type) (b Type)) (= (shape-of (type-fn a b)) SHAPE_ARROW)))\n");
        // Tuple2 has shape TUPLE
        smt.push_str("(assert (forall ((a Type) (b Type)) (= (shape-of (type-tuple2 a b)) SHAPE_TUPLE)))\n");
        // Struct has shape CONSTRUCTOR
        smt.push_str("(assert (forall ((tag Int) (a Type)) (= (shape-of (type-struct tag a)) SHAPE_CONSTRUCTOR)))\n");
        // Poly container
        smt.push_str("(assert (forall ((a Type)) (= (shape-of (type-poly a)) SHAPE_POLY)))\n\n");

        // ── 6. Declare inference variables as constants ────────────
        // We only declare the target variable; other variables are
        // represented implicitly through the constraints they appear in.
        smt.push_str(&format!("(declare-const iv_{} Type)\n", var_id));

        // ── 7. Inject the solver context (constraints) ─────────────
        // The caller provides encoded Eq/Sub constraints.
        smt.push_str("; Active constraints from the solver context\n");
        smt.push_str(solver_ctx);
        smt.push('\n');

        // ── 8. For each candidate shape, check if it's forced ──────
        let shape_names = [
            ("SHAPE_UNKNOWN", PrincipalShape::Unknown),
            ("SHAPE_ARROW", PrincipalShape::Arrow),
            ("SHAPE_TUPLE", PrincipalShape::Tuple(2)),
            ("SHAPE_CONSTRUCTOR", PrincipalShape::Constructor(0)),
            ("SHAPE_POLY", PrincipalShape::Poly),
        ];

        let mut possible_shapes: Vec<PrincipalShape> = Vec::new();

        for (name, ps) in &shape_names {
            smt.push_str(&format!(
                "(push 1)\n\
                 (assert (= (shape-of iv_{}) {}))\n\
                 (check-sat)\n\
                 (pop 1)\n",
                var_id, name
            ));
        }

        // ── 9. Query Z3 ─────────────────────────────────────────
        let output = self.call_z3(&smt)?;
        self.parse_unicity_results(&output, &shape_names)
    }

    /// Encode a set of equality constraints as SMT-LIB2 formulas.
    /// `eq_constraints` is a list of (lhs_id, rhs_id) pairs where each
    /// id refers to an InferVar. Concrete types on either side are
    /// represented by their SMT constructor term.
    pub fn encode_eq_constraints(
        &self,
        ctx: &TypeContext,
        eq_constraints: &[(usize, usize)],
    ) -> String {
        let mut smt = String::new();
        for (a_id, b_id) in eq_constraints {
            smt.push_str(&format!(
                "(assert (= iv_{} iv_{}))\n",
                a_id, b_id
            ));
        }
        smt
    }

    /// Encode bindings (InferVar → resolved TypeId) as SMT equalities.
    pub fn encode_bindings(
        &self,
        ctx: &TypeContext,
        bindings: &HashMap<usize, TypeId>,
    ) -> String {
        let mut smt = String::new();
        for (var_id, bound_ty) in bindings {
            let term = self.type_to_smt_term(ctx, *bound_ty);
            smt.push_str(&format!("(assert (= iv_{} {}))\n", var_id, term));
        }
        smt
    }

    /// Convert a resolved TypeId to an SMT-LIB2 term.
    fn type_to_smt_term(&self, ctx: &TypeContext, ty: TypeId) -> String {
        let resolved = ctx.resolve_binding(ty);
        match ctx.get(resolved) {
            TypeData::Int { bits, signed } => {
                if *bits == 32 { "type-int32".into() }
                else { "type-int64".into() }
            }
            TypeData::Bool => "type-bool".into(),
            TypeData::Unit => "type-unit".into(),
            TypeData::Never => "type-never".into(),
            TypeData::Char => "type-char".into(),
            TypeData::Byte => "type-byte".into(),
            TypeData::Fn { params, ret } => {
                if params.len() == 1 {
                    let p = self.type_to_smt_term(ctx, params[0]);
                    let r = self.type_to_smt_term(ctx, *ret);
                    format!("(type-fn {} {})", p, r)
                } else if params.len() == 2 {
                    let p1 = self.type_to_smt_term(ctx, params[0]);
                    let p2 = self.type_to_smt_term(ctx, params[1]);
                    let r = self.type_to_smt_term(ctx, *ret);
                    // Encode multi-arg fn as nested single-arg fn
                    format!("(type-fn {} (type-fn {} {}))", p1, p2, r)
                } else {
                    format!("type-unknown")
                }
            }
            TypeData::Coproduct { .. } | TypeData::Tuple { .. } => {
                // Simplified: just use tuple2 for any structure
                "type-unknown".into()
            }
            TypeData::Forall { body, .. } | TypeData::Exists { base: body, .. }
            | TypeData::Poly { body, .. } => {
                let b = self.type_to_smt_term(ctx, *body);
                format!("(type-poly {})", b)
            }
            TypeData::Mu { body, .. } | TypeData::Nu { body, .. } => {
                let b = self.type_to_smt_term(ctx, *body);
                format!("(type-poly {})", b)
            }
            TypeData::InferVar { id } => {
                format!("iv_{}", id)
            }
            _ => "type-unknown".into(),
        }
    }

    fn call_z3(&self, smt: &str) -> Option<String> {
        if smt.is_empty() {
            return None;
        }
        let mut child = Command::new(&self.z3_path)
            .arg("-in")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .ok()?;

        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(smt.as_bytes()).ok()?;
        }

        let output = child.wait_with_output().ok()?;
        if output.status.success() {
            Some(String::from_utf8_lossy(&output.stdout).to_string())
        } else {
            None
        }
    }

    fn parse_unicity_results(
        &self,
        output: &str,
        shape_names: &[(&str, PrincipalShape)],
    ) -> Option<PrincipalShape> {
        let mut unique_shape: Option<PrincipalShape> = None;
        for (i, line) in output.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed == "sat" && i < shape_names.len() {
                // This candidate shape is consistent with all constraints.
                // If we've already found one consistent shape, unicity fails.
                if unique_shape.is_some() {
                    return None; // multiple shapes possible
                }
                unique_shape = Some(shape_names[i].1.clone());
            }
        }
        unique_shape
    }
}
