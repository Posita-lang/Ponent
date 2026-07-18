//! # Defaulting Logic — Resolve Unconstrained Inference Variables
//!
//! Extracted from `InferenceContext::solve` into a standalone function
//! so that both the old solver (`InferenceContext::solve`) and the new
//! solver (`FulfillmentContext::evaluate_all`) can call it without
//! depending on the full `InferenceContext`.
//!
//! ## Design
//!
//! The defaulting runs in TWO passes:
//! 1. Pass 1 defaults "guided" kinds (Integer → Int<32>, Float → Float<64>,
//!    Bool → Bool, Numeric → Int<32>).  This resolves the leaf variables
//!    first, so that Any/Unconstrained variables unified with them are
//!    transitively resolved.
//! 2. Pass 2 handles Any/Unconstrained variables that are still unresolved:
//!    - Expression(Some(span)) → Err(CannotInfer) — inference failure
//!    - Expression(None) → silently default to Error
//!    - GenericParam → silently default to Error
//!    - Synthetic → silently default to Error

use crate::hir::infer::{GenStatus, TypeVariableKind, VarOrigin};
use crate::hir::types::{TypeContext, TypeData, TypeError, TypeId};

/// Default unresolved inference variables to their default types.
///
/// # Arguments
/// * `ctx` — The type context (for resolving bindings and setting new ones).
/// * `var_type_ids` — The `TypeId` for each inference variable (indexed by var_id).
/// * `type_vars` — The `TypeVar` metadata for each variable (kind, shape, etc.).
/// * `gen_statuses` — The generalisation status for each variable.
/// * `var_origins` — The origin of each variable (for error reporting).
///
/// # Returns
/// * `Ok(())` — All variables were successfully defaulted or skipped.
/// * `Err(TypeError::CannotInfer { span })` — An `Any`/`Unconstrained` variable
///   with `Expression(Some(span))` origin could not be resolved.
pub fn default_variables(
    ctx: &mut TypeContext,
    var_type_ids: &[TypeId],
    type_vars: &[(TypeVariableKind, VarOrigin)],
    gen_statuses: &[GenStatus],
) -> Result<(), TypeError> {
    // ── Pass 1: default guided kinds (Integer, Float, Bool, Numeric) ──
    for (i, &ty_id) in var_type_ids.iter().enumerate() {
        let resolved = ctx.resolve_binding(ty_id);
        if let TypeData::InferVar { .. } = ctx.get(resolved) {
            // Skip PartiallyGeneralizable variables — they are guarded by
            // suspended constraints and will be resolved later.
            if i < gen_statuses.len() && gen_statuses[i] == GenStatus::PartiallyGeneralizable {
                continue;
            }
            let (kind, _origin) = type_vars
                .get(i)
                .copied()
                .unwrap_or((TypeVariableKind::Any, VarOrigin::Synthetic));
            match kind {
                TypeVariableKind::Integer => {
                    let default_ty = ctx.int(32, true);
                    ctx.set_binding(ty_id, default_ty);
                }
                TypeVariableKind::Float => {
                    let default_ty = ctx.float(64);
                    ctx.set_binding(ty_id, default_ty);
                }
                TypeVariableKind::Bool => {
                    let default_ty = ctx.bool();
                    ctx.set_binding(ty_id, default_ty);
                }
                TypeVariableKind::Numeric => {
                    let default_ty = ctx.int(32, true);
                    ctx.set_binding(ty_id, default_ty);
                }
                _ => {} // Unconstrained / Any handled in Pass 2
            }
        }
    }

    // ── Pass 2: check Unconstrained / Any for Expression errors ─────
    for (i, &ty_id) in var_type_ids.iter().enumerate() {
        let resolved = ctx.resolve_binding(ty_id);
        if let TypeData::InferVar { .. } = ctx.get(resolved) {
            if i < gen_statuses.len() && gen_statuses[i] == GenStatus::PartiallyGeneralizable {
                continue;
            }
            let (kind, origin) = type_vars
                .get(i)
                .copied()
                .unwrap_or((TypeVariableKind::Any, VarOrigin::Synthetic));
            // Expression-level Unconstrained/Any that was never resolved
            // is a type inference failure — report it.
            if matches!(
                kind,
                TypeVariableKind::Unconstrained | TypeVariableKind::Any
            ) {
                if let VarOrigin::Expression(Some(span)) = origin {
                    return Err(TypeError::CannotInfer { span });
                }
            }
            let default_ty = match kind {
                TypeVariableKind::Unconstrained => ctx.error(),
                TypeVariableKind::Any => ctx.error(),
                _ => continue,
            };
            ctx.set_binding(ty_id, default_ty);
        }
    }

    Ok(())
}
