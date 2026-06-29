use crate::ast::Span;
use crate::hir::symbol::SymbolTable;
use crate::hir::traits::TraitEnv;
use crate::hir::types::*;
use rustc_hash::FxHashMap as HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeVariableKind {
    Unconstrained,
    Integer,
    Float,
    Numeric,
    Bool,
    Any,
}

#[derive(Debug, Clone)]
pub struct TypeVar {
    pub id: usize,
    pub kind: TypeVariableKind,
}

#[derive(Debug, Clone)]
pub enum Constraint {
    Eq(TypeId, TypeId, Span),
    Sub(TypeId, TypeId, Span),
    Impl(TypeId, DefId, Span),
}

pub struct InferenceContext {
    type_vars: Vec<TypeVar>,
    var_type_ids: Vec<TypeId>,
    constraints: Vec<Constraint>,
    next_var_id: usize,
    /// Per-variable lower bounds (subtypes that must be ≤ this variable).
    /// lower_bounds[i] contains TypeIds that must be subtypes of variable i.
    lower_bounds: Vec<Vec<TypeId>>,
    /// Per-variable upper bounds (supertypes that this variable must be ≤).
    /// upper_bounds[i] contains TypeIds that variable i must be a subtype of.
    upper_bounds: Vec<Vec<TypeId>>,
}

impl InferenceContext {
    pub fn new() -> Self {
        InferenceContext {
            type_vars: Vec::new(),
            var_type_ids: Vec::new(),
            constraints: Vec::new(),
            next_var_id: 0,
            lower_bounds: Vec::new(),
            upper_bounds: Vec::new(),
        }
    }

    pub fn new_type_var(&mut self, ctx: &mut TypeContext, kind: TypeVariableKind) -> TypeId {
        let id = self.next_var_id;
        self.next_var_id += 1;
        let ty_id = ctx.alloc_infer_var(id);
        self.type_vars.push(TypeVar { id, kind });
        self.var_type_ids.push(ty_id);
        // Grow bounds vectors to match the new variable id
        while self.lower_bounds.len() <= id {
            self.lower_bounds.push(Vec::new());
        }
        while self.upper_bounds.len() <= id {
            self.upper_bounds.push(Vec::new());
        }
        ty_id
    }

    /// Look up the kind of a type variable by its id.
    pub fn get_var_kind(&self, id: usize) -> Option<TypeVariableKind> {
        self.type_vars.iter().find(|tv| tv.id == id).map(|tv| tv.kind)
    }

    pub fn add_constraint(&mut self, c: Constraint) {
        self.constraints.push(c);
    }

    pub fn solve(&mut self, ctx: &mut TypeContext, trait_env: &TraitEnv, symbols: &SymbolTable) -> Result<(), TypeError> {
        // First, solve all Eq constraints
        for c in &self.constraints {
            if let Constraint::Eq(a, b, _) = c {
                ctx.unify(*a, *b)?;
            }
        }

        // Then process Sub constraints: record bounds for inference variables
        for c in &self.constraints {
            if let Constraint::Sub(sub, sup, _span) = c {
                let resolved_sub = ctx.resolve_binding(*sub);
                let resolved_sup = ctx.resolve_binding(*sup);

                // If sup is an InferVar, record sub as a lower bound of sup
                if let TypeData::InferVar { id } = ctx.get(resolved_sup) {
                    if *id < self.lower_bounds.len() {
                        self.lower_bounds[*id].push(resolved_sub);
                    }
                }
                // If sub is an InferVar, record sup as an upper bound of sub
                if let TypeData::InferVar { id } = ctx.get(resolved_sub) {
                    if *id < self.upper_bounds.len() {
                        self.upper_bounds[*id].push(resolved_sup);
                    }
                }

                // If both sides are resolved (not InferVar), check the subtype relationship now
                let sub_is_infer = matches!(ctx.get(resolved_sub), TypeData::InferVar { .. });
                let sup_is_infer = matches!(ctx.get(resolved_sup), TypeData::InferVar { .. });
                if !sub_is_infer && !sup_is_infer {
                    if !ctx.subtype(resolved_sub, resolved_sup) {
                        return Err(TypeError::Mismatch {
                            expected: resolved_sup,
                            found: resolved_sub,
                            span: *_span,
                        });
                    }
                }
            }
        }

        // Then check Impl constraints
        for c in &self.constraints {
            if let Constraint::Impl(ty, trait_id, span) = c {
                let resolved = ctx.resolve_binding(*ty);
                let data = ctx.get(resolved);
                // If the type is an error, skip
                if matches!(data, TypeData::Error) {
                    continue;
                }
                // If still an infer var, that's fine; solving will assign a default later
                if matches!(data, TypeData::InferVar { .. }) {
                    continue;
                }
                // Otherwise, check that the impl exists
                let impl_found = if trait_env.lookup_impl(*trait_id, resolved).is_some() {
                    true
                } else {
                    // Try generic matching when exact match fails
                    trait_env.lookup_impl_generic(*trait_id, resolved, ctx, symbols).is_some()
                };
                if !impl_found {
                    return Err(TypeError::TraitNotImplemented {
                        ty: *ty,
                        trait_name: format!("{:?}", trait_id),
                        span: *span,
                    });
                }
                // Generate obligations for associated types: when we have a
                // resolved Impl(concrete_ty, trait_id, _), look for concrete types
                // for any AssociatedType { trait_id, name, self_ty } by matching
                // the impl's assoc_tys entries.
                if let Some(impl_candidate) = trait_env.lookup_impl(*trait_id, resolved) {
                    for (assoc_name, assoc_ty) in &impl_candidate.assoc_tys {
                        // Walk all Eq constraints to substitute any AssociatedType
                        // that matches this name, trait_id, and self_ty
                        for eq_c in &self.constraints {
                            if let Constraint::Eq(a, b, _) = eq_c {
                                for id in &[*a, *b] {
                                    let resolved_id = ctx.resolve_binding(*id);
                                    if let TypeData::AssociatedType {
                                        trait_id: at_trait_id,
                                        name: at_name,
                                        self_ty: at_self,
                                    } = ctx.get(resolved_id).clone()
                                    {
                                        if at_trait_id == *trait_id
                                            && at_name == *assoc_name
                                            && ctx.resolve_binding(at_self) == resolved
                                        {
                                            ctx.unify(resolved_id, *assoc_ty)?;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Kind checking: ensure that solved types respect the variable's kind
        for (i, &ty_id) in self.var_type_ids.iter().enumerate() {
            let resolved = ctx.resolve_binding(ty_id);
            let data = ctx.get(resolved);
            if let TypeData::InferVar { .. } = data {
                continue; // will be defaulted below
            }
            if matches!(data, TypeData::Error) {
                continue;
            }
            let kind = self.type_vars[i].kind;
            match kind {
                TypeVariableKind::Integer => {
                    if !matches!(
                        data,
                        TypeData::Int { .. } | TypeData::UInt { .. } | TypeData::USize
                    ) {
                        return Err(TypeError::Mismatch {
                            expected: ty_id,
                            found: ty_id,
                            span: Span::new(0, 0),
                        });
                    }
                }
                TypeVariableKind::Float => {
                    if !matches!(data, TypeData::Float { .. }) {
                        return Err(TypeError::Mismatch {
                            expected: ty_id,
                            found: ty_id,
                            span: Span::new(0, 0),
                        });
                    }
                }
                TypeVariableKind::Bool => {
                    if !matches!(data, TypeData::Bool) {
                        return Err(TypeError::Mismatch {
                            expected: ty_id,
                            found: ty_id,
                            span: Span::new(0, 0),
                        });
                    }
                }
                TypeVariableKind::Numeric => {
                    if !matches!(
                        data,
                        TypeData::Int { .. }
                            | TypeData::UInt { .. }
                            | TypeData::Float { .. }
                            | TypeData::USize
                    ) {
                        return Err(TypeError::Mismatch {
                            expected: ty_id,
                            found: ty_id,
                            span: Span::new(0, 0),
                        });
                    }
                }
                _ => {}
            }
        }

        // Defaulting: unfilled infer vars get default types
        for (i, &ty_id) in self.var_type_ids.iter().enumerate() {
            let resolved = ctx.resolve_binding(ty_id);
            if let TypeData::InferVar { .. } = ctx.get(resolved) {
                let default_ty = match self.type_vars[i].kind {
                    TypeVariableKind::Integer => ctx.int(32, true),
                    TypeVariableKind::Float => ctx.float(64),
                    TypeVariableKind::Bool => ctx.bool(),
                    TypeVariableKind::Numeric => ctx.int(32, true),
                    TypeVariableKind::Unconstrained => ctx.error(),
                    TypeVariableKind::Any => ctx.error(),
                };
                ctx.bindings.borrow_mut().insert(ty_id, default_ty);
            }
        }

        Ok(())
    }

    pub fn finalize(&self, ctx: &mut TypeContext) -> HashMap<usize, TypeId> {
        let mut solution = HashMap::default();
        for (i, &ty_id) in self.var_type_ids.iter().enumerate() {
            let resolved = ctx.resolve_binding(ty_id);
            let data = ctx.get(resolved);
            match data {
                TypeData::InferVar { id } => {
                    // Variable is still unbound — try to infer from bounds
                    let var_id = *id;
                    let lbs: &[TypeId] = if var_id < self.lower_bounds.len() {
                        &self.lower_bounds[var_id]
                    } else {
                        &[]
                    };
                    let ubs: &[TypeId] = if var_id < self.upper_bounds.len() {
                        &self.upper_bounds[var_id]
                    } else {
                        &[]
                    };
                    let chosen = if !lbs.is_empty() {
                        // Covariant: pick the least upper bound from lower bounds
                        // (simple heuristic: pick the first resolved lower bound)
                        let first_resolved = lbs.iter().find(|t| {
                            let r = ctx.resolve_binding(**t);
                            !matches!(ctx.get(r), TypeData::InferVar { .. })
                        });
                        first_resolved.copied().unwrap_or(ctx.error())
                    } else if !ubs.is_empty() {
                        // Contravariant: pick the greatest lower bound from upper bounds
                        let first_resolved = ubs.iter().find(|t| {
                            let r = ctx.resolve_binding(**t);
                            !matches!(ctx.get(r), TypeData::InferVar { .. })
                        });
                        first_resolved.copied().unwrap_or(ctx.error())
                    } else {
                        // No bounds — default based on kind
                        match self.type_vars[i].kind {
                            TypeVariableKind::Integer => ctx.int(32, true),
                            TypeVariableKind::Float => ctx.float(64),
                            TypeVariableKind::Bool => ctx.bool(),
                            TypeVariableKind::Numeric => ctx.int(32, true),
                            _ => ctx.error(),
                        }
                    };
                    solution.insert(var_id, chosen);
                }
                _ => {
                    solution.insert(self.type_vars[i].id, resolved);
                }
            }
        }
        solution
    }

    pub fn apply_solution(
        ty: TypeId,
        solution: &HashMap<usize, TypeId>,
        ctx: &TypeContext,
    ) -> TypeId {
        replace_infer(ty, solution, ctx)
    }
}

impl Default for InferenceContext {
    fn default() -> Self {
        Self::new()
    }
}

fn replace_infer(ty: TypeId, solution: &HashMap<usize, TypeId>, ctx: &TypeContext) -> TypeId {
    let resolved = ctx.resolve_binding(ty);
    let data = ctx.get(resolved).clone();
    match data {
        TypeData::InferVar { id } => solution.get(&id).copied().unwrap_or(ty),
        TypeData::Int { .. }
        | TypeData::UInt { .. }
        | TypeData::Float { .. }
        | TypeData::Bool
        | TypeData::Char
        | TypeData::Byte
        | TypeData::USize
        | TypeData::Never
        | TypeData::Unit
        | TypeData::Error => ty,
        TypeData::GenericParam { .. } => ty,
        TypeData::Struct { def_id, args } => {
            let new_args: Vec<TypeId> = args
                .iter()
                .map(|&a| replace_infer(a, solution, ctx))
                .collect();
            ctx.find_type(&TypeData::Struct {
                def_id,
                args: new_args,
            })
            .unwrap_or(ctx.error())
        }
        TypeData::Enum { def_id, args } => {
            let new_args: Vec<TypeId> = args
                .iter()
                .map(|&a| replace_infer(a, solution, ctx))
                .collect();
            ctx.find_type(&TypeData::Enum {
                def_id,
                args: new_args,
            })
            .unwrap_or(ctx.error())
        }
        TypeData::Tuple { elems } => {
            let new_elems: Vec<TypeId> = elems
                .iter()
                .map(|&e| replace_infer(e, solution, ctx))
                .collect();
            ctx.find_type(&TypeData::Tuple { elems: new_elems })
                .unwrap_or(ctx.error())
        }
        TypeData::Array { elem, size } => {
            let new_elem = replace_infer(elem, solution, ctx);
            ctx.find_type(&TypeData::Array {
                elem: new_elem,
                size,
            })
            .unwrap_or(ctx.error())
        }
        TypeData::Slice { elem } => {
            let new_elem = replace_infer(elem, solution, ctx);
            ctx.find_type(&TypeData::Slice { elem: new_elem })
                .unwrap_or(ctx.error())
        }
        TypeData::Ref { ty, mutable } => {
            let new_ty = replace_infer(ty, solution, ctx);
            ctx.find_type(&TypeData::Ref {
                ty: new_ty,
                mutable,
            })
            .unwrap_or(ctx.error())
        }
        TypeData::Pointer { ty } => {
            let new_ty = replace_infer(ty, solution, ctx);
            ctx.find_type(&TypeData::Pointer { ty: new_ty })
                .unwrap_or(ctx.error())
        }
        TypeData::Ptr { size, pointee } => {
            let new_size = replace_infer(size, solution, ctx);
            let new_pointee = replace_infer(pointee, solution, ctx);
            ctx.find_type(&TypeData::Ptr {
                size: new_size,
                pointee: new_pointee,
            })
            .unwrap_or(ctx.error())
        }
        TypeData::Fn { params, ret } => {
            let new_params: Vec<TypeId> = params
                .iter()
                .map(|&p| replace_infer(p, solution, ctx))
                .collect();
            let new_ret = replace_infer(ret, solution, ctx);
            ctx.find_type(&TypeData::Fn {
                params: new_params,
                ret: new_ret,
            })
            .unwrap_or(ctx.error())
        }
        TypeData::DynTrait { .. } => ty,
        TypeData::Exists { name, base } => {
            let new_base = replace_infer(base, solution, ctx);
            ctx.find_type(&TypeData::Exists {
                name,
                base: new_base,
            })
            .unwrap_or(ctx.error())
        }
        TypeData::AssociatedType {
            trait_id,
            name,
            self_ty,
        } => {
            let new_self = replace_infer(self_ty, solution, ctx);
            ctx.find_type(&TypeData::AssociatedType {
                trait_id,
                name,
                self_ty: new_self,
            })
            .unwrap_or(ctx.error())
        }
        TypeData::Forall { param_index, param_name, body } => {
            let new_body = replace_infer(body, solution, ctx);
            ctx.find_type(&TypeData::Forall { param_index, param_name, body: new_body })
                .unwrap_or(ctx.error())
        }
    }
}
