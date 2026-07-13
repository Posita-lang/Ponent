use crate::ast::*;
use crate::diagnostics::Diagnostic;
use crate::hir::types::*;
use crate::symbol::Symbol;
use indexmap::IndexMap;
use rustc_hash::FxBuildHasher;
use rustc_hash::FxHashMap as HashMap;

#[derive(Debug, Clone)]
pub struct FieldBinding {
    pub name: Symbol,
    pub ty: TypeId,
    pub default: Option<Expr>,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TypeKind {
    Struct,
    Enum,
    Alias,
    Trait,
    Impl,
    Constraint,
}

#[derive(Debug, Clone)]
pub struct VariableBinding {
    pub ty: TypeId,
    pub mutable: bool,
    pub span: Span,
    pub def_id: DefId,
}

#[derive(Debug, Clone)]
pub struct Parameter {
    pub name: Symbol,
    pub ty: TypeId,
    pub span: Span,
    pub default: Option<Expr>,
}

#[derive(Debug, Clone)]
pub struct FunctionSignature {
    pub params: Vec<Parameter>,
    pub return_type: TypeId,
    pub type_params: Vec<TypeParam>,
    pub where_clause: Option<WhereClause>,
}

#[derive(Debug, Clone)]
pub struct FunctionBinding {
    pub def_id: DefId,
    pub signature: FunctionSignature,
    pub is_comptime: bool,
    pub is_async: bool,
    pub is_pure: bool,
    pub is_ieee_contracts: bool,
    pub contracts: Vec<Contract>,
    pub hints: Vec<Expr>,
    pub attributes: Vec<Attribute>,
}

#[derive(Debug, Clone)]
pub struct TypeBinding {
    pub def_id: DefId,
    pub params: Vec<TypeParam>,
    pub kind: TypeKind,
    pub span: Span,
    pub alias_ast: Option<Type>,
    pub fields: Vec<FieldBinding>,
    pub variants: Vec<EnumVariant>,
    pub invariant: Option<Expr>,
    pub default_value: Option<Expr>,
    pub no_default: bool,
    pub crate_id: CrateId,
    /// Custom error message for non-exhaustive match on this type.
    /// Set by `with missing_match = "..."` on enum definitions.
    pub missing_match: Option<String>,
    /// If true, all `match`, `if let`, and `while let` on this type
    /// must be exhaustive — `_` wildcards are forbidden.
    /// Set by `@exhaustive` attribute on the type.
    pub exhaustive: bool,
    /// Layout representation hints (from `@repr` attributes).
    pub c_layout: bool,
    /// If true, this single-field type has the same layout as its sole field.
    /// Set by `@transparent` attribute.
    pub transparent: bool,
    /// Expanded layout attributes from `@layout(AliasName)` resolution.
    /// Contains the full set of built-in attributes (packed, endian, etc.)
    /// after alias expansion, for use by codegen / layout_of!.
    pub expanded_layout_attrs: Vec<crate::ast::Attribute>,
    /// Whether `@packed` is set on this type (remove padding between fields).
    pub packed: bool,
    /// Endianness from `@endian(little)` or `@endian(big)`.
    pub endian: Option<crate::ast::Endianness>,
    /// Bit field fill order from `@bit_order(lsb_to_msb)` or `@bit_order(msb_to_lsb)`.
    pub bit_order: Option<crate::ast::BitOrder>,
    /// Alignment override from `@align(N)`.
    pub align: Option<u64>,
    /// Padding from `@pad(N)`.
    pub pad: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct TraitBinding {
    pub def_id: DefId,
    pub methods: Vec<(Symbol, FunctionSignature)>,
    pub associated_types: Vec<(Symbol, Option<Type>)>,
    pub span: Span,
    pub crate_id: CrateId,
}

#[derive(Debug, Clone)]
pub struct ImplBinding {
    pub def_id: DefId,
    pub methods: Vec<ImplMethod>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct ConstraintPredicate {
    pub subject: TypeId,
    pub bounds: Vec<TypeId>,
}

#[derive(Debug, Clone)]
pub struct ConstraintBinding {
    /// Each predicate is a subject type with its trait bounds,
    /// matching the syntax `Subject: Bound1 + Bound2`.
    pub predicates: Vec<ConstraintPredicate>,
    /// The type parameters declared on the constraint (e.g. `<T>` in
    /// `constraint Foo<T> { T: Display }`).  Stored so the checker can build
    /// a substitution map when the constraint is instantiated at a use site.
    pub params: Vec<crate::ast::TypeParam>,
    pub span: Span,
}

pub struct Scope {
    pub parent: Option<usize>,
    pub variables: IndexMap<Symbol, VariableBinding, FxBuildHasher>,
    pub functions: HashMap<Symbol, FunctionBinding>,
    pub types: HashMap<Symbol, TypeBinding>,
    pub traits: HashMap<Symbol, TraitBinding>,
    pub impls: Vec<ImplBinding>,
    pub constraints: HashMap<Symbol, ConstraintBinding>,
    /// If true, duplicate names in this scope are allowed (shadowing).
    /// Block scopes are ordered; module/function scopes are not.
    pub ordered: bool,
}

impl Scope {
    pub fn new(parent: Option<usize>, ordered: bool) -> Self {
        Scope {
            parent,
            variables: IndexMap::with_hasher(FxBuildHasher::default()),
            functions: HashMap::default(),
            types: HashMap::default(),
            traits: HashMap::default(),
            impls: Vec::new(),
            constraints: HashMap::default(),
            ordered,
        }
    }
}

pub struct SymbolTable {
    scopes: Vec<Scope>,
    current_scope: usize,
    type_defs: HashMap<DefId, TypeBinding>,
    trait_defs: HashMap<DefId, TraitBinding>,
    next_def_id: usize,
    pub local_crate_id: CrateId,
    /// Maps fully-qualified type path (e.g. "std::collections::HashMap") to DefId.
    /// Used to resolve multi-segment paths when no module hierarchy exists yet.
    full_path_to_def_id: HashMap<Symbol, DefId>,
}

impl SymbolTable {
    pub fn new(local_crate_id: CrateId) -> Self {
        let root = Scope::new(None, false);
        SymbolTable {
            scopes: vec![root],
            current_scope: 0,
            type_defs: HashMap::default(),
            trait_defs: HashMap::default(),
            next_def_id: 0,
            local_crate_id,
            full_path_to_def_id: HashMap::default(),
        }
    }

    pub fn push_scope(&mut self) -> usize {
        let parent = Some(self.current_scope);
        let scope = Scope::new(parent, true); // block scopes are ordered (allow shadowing)
        self.scopes.push(scope);
        self.current_scope = self.scopes.len() - 1;
        self.current_scope
    }

    pub fn pop_scope(&mut self) {
        if let Some(parent) = self.scopes[self.current_scope].parent {
            self.current_scope = parent;
        }
    }

    pub fn current_scope(&self) -> usize {
        self.current_scope
    }

    pub fn insert_variable(
        &mut self,
        name: Symbol,
        binding: VariableBinding,
        span: Span,
    ) -> Result<(), Diagnostic> {
        let scope = &mut self.scopes[self.current_scope];
        if !scope.ordered && scope.variables.contains_key(&name) {
            return Err(
                Diagnostic::error(format!("variable '{}' already defined", name.as_str())).with_span(span),
            );
        }
        scope.variables.insert(name, binding);
        Ok(())
    }

    pub fn insert_function(
        &mut self,
        name: Symbol,
        binding: FunctionBinding,
        span: Span,
    ) -> Result<(), Diagnostic> {
        let scope = &mut self.scopes[self.current_scope];
        if scope.functions.contains_key(&name) {
            return Err(
                Diagnostic::error(format!("function '{}' already defined", name.as_str())).with_span(span),
            );
        }
        scope.functions.insert(name, binding);
        Ok(())
    }

    pub fn insert_type(
        &mut self,
        name: Symbol,
        binding: TypeBinding,
        span: Span,
    ) -> Result<(), Diagnostic> {
        let scope = &mut self.scopes[self.current_scope];
        if scope.types.contains_key(&name) {
            return Err(
                Diagnostic::error(format!("type '{}' already defined", name.as_str())).with_span(span),
            );
        }
        let def_id = binding.def_id;
        self.type_defs.insert(def_id, binding.clone());
        scope.types.insert(name, binding);
        // Note: intentionally NOT inserting the simple name into
        // full_path_to_def_id here.  The full-path map is reserved for
        // fully-qualified paths (e.g. "std::collections::HashMap")
        // registered via register_full_path.  Inserting simple names
        // would cause lookup_type_by_path to bypass lexical scoping.
        Ok(())
    }

    /// Register a fully-qualified type path (e.g. "std::collections::HashMap")
    /// mapping to an already-inserted DefId.  Used by the resolver when it
    /// encounters type definitions inside modules or when processing imports.
    pub fn register_full_path(&mut self, full_path: Symbol, def_id: DefId) {
        self.full_path_to_def_id.entry(full_path).or_insert(def_id);
    }

    pub fn insert_trait(
        &mut self,
        name: Symbol,
        binding: TraitBinding,
        span: Span,
    ) -> Result<(), Diagnostic> {
        let scope = &mut self.scopes[self.current_scope];
        if scope.traits.contains_key(&name) {
            return Err(
                Diagnostic::error(format!("trait '{}' already defined", name.as_str())).with_span(span),
            );
        }
        let def_id = binding.def_id;
        self.trait_defs.insert(def_id, binding.clone());
        scope.traits.insert(name, binding);
        Ok(())
    }

    pub fn insert_impl(&mut self, binding: ImplBinding, span: Span) {
        let scope = &mut self.scopes[self.current_scope];
        scope.impls.push(binding);
    }

    pub fn insert_constraint(
        &mut self,
        name: Symbol,
        binding: ConstraintBinding,
        span: Span,
    ) -> Result<(), Diagnostic> {
        let scope = &mut self.scopes[self.current_scope];
        if scope.constraints.contains_key(&name) {
            return Err(
                Diagnostic::error(format!("constraint '{}' already defined", name.as_str())).with_span(span),
            );
        }
        scope.constraints.insert(name, binding);
        Ok(())
    }

    pub fn lookup_variable(&self, name: Symbol, span: Span) -> Option<&VariableBinding> {
        let mut idx = self.current_scope;
        while let Some(scope) = self.scopes.get(idx) {
            if let Some(binding) = scope.variables.get(&name) {
                return Some(binding);
            }
            if let Some(parent) = scope.parent {
                idx = parent;
            } else {
                break;
            }
        }
        None
    }

    pub fn lookup_function(&self, name: Symbol) -> Option<&FunctionBinding> {
        let mut idx = self.current_scope;
        while let Some(scope) = self.scopes.get(idx) {
            if let Some(binding) = scope.functions.get(&name) {
                return Some(binding);
            }
            if let Some(parent) = scope.parent {
                idx = parent;
            } else {
                break;
            }
        }
        None
    }

    pub fn lookup_type(&self, name: Symbol) -> Option<&TypeBinding> {
        let mut idx = self.current_scope;
        while let Some(scope) = self.scopes.get(idx) {
            if let Some(binding) = scope.types.get(&name) {
                return Some(binding);
            }
            if let Some(parent) = scope.parent {
                idx = parent;
            } else {
                break;
            }
        }
        None
    }

    pub fn lookup_type_by_def_id(&self, def_id: DefId) -> Option<&TypeBinding> {
        self.type_defs.get(&def_id)
    }

    pub fn lookup_trait(&self, name: Symbol) -> Option<&TraitBinding> {
        let mut idx = self.current_scope;
        while let Some(scope) = self.scopes.get(idx) {
            if let Some(binding) = scope.traits.get(&name) {
                return Some(binding);
            }
            if let Some(parent) = scope.parent {
                idx = parent;
            } else {
                break;
            }
        }
        None
    }

    pub fn lookup_trait_by_def_id(&self, def_id: DefId) -> Option<&TraitBinding> {
        self.trait_defs.get(&def_id)
    }

    /// Expose the full-path-to-DefId table for multi-segment lookups.
    pub fn full_path_to_def_id(&self) -> &HashMap<Symbol, DefId> {
        &self.full_path_to_def_id
    }

    pub fn lookup_constraint(&self, name: Symbol) -> Option<&ConstraintBinding> {
        let mut idx = self.current_scope;
        while let Some(scope) = self.scopes.get(idx) {
            if let Some(binding) = scope.constraints.get(&name) {
                return Some(binding);
            }
            if let Some(parent) = scope.parent {
                idx = parent;
            } else {
                break;
            }
        }
        None
    }

    pub fn lookup_type_by_path(&self, path: &[Symbol]) -> Option<DefId> {
        if path.is_empty() {
            return None;
        }
        // For single-segment paths, use scoped lookup directly — the
        // full-path map is reserved for multi-segment qualified paths.
        if path.len() == 1 {
            let binding = self.lookup_type(path[0])?;
            return Some(binding.def_id);
        }
        // Multi-segment path: try the full-path cache first.
        let full = path
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join("::");
        let full_sym = Symbol::intern(&full);
        if let Some(&id) = self.full_path_to_def_id.get(&full_sym) {
            return Some(id);
        }
        None
    }

    pub fn lookup_trait_by_path(&self, path: &[Symbol]) -> Option<DefId> {
        // Try the full-path lookup first.
        let full = path
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join("::");
        let full_sym = Symbol::intern(&full);
        if let Some(&id) = self.full_path_to_def_id.get(&full_sym) {
            // Check if this DefId is actually a trait (by looking it up in trait_defs)
            if self.trait_defs.contains_key(&id) {
                return Some(id);
            }
        }
        if path.len() != 1 {
            return None;
        }
        let binding = self.lookup_trait(path[0])?;
        Some(binding.def_id)
    }

    /// Find all traits that define an associated type with the given name.
    pub fn lookup_traits_by_assoc_type_name(&self, name: Symbol) -> Vec<DefId> {
        self.trait_defs
            .iter()
            .filter(|(_, b)| b.associated_types.iter().any(|(n, _)| *n == name))
            .map(|(id, _)| *id)
            .collect()
    }

    pub fn lookup_field(&self, def_id: DefId, name: Symbol) -> Option<TypeId> {
        let binding = self.type_defs.get(&def_id)?;
        if let TypeKind::Struct = binding.kind {
            binding.fields.iter().find(|f| f.name == name).map(|f| f.ty)
        } else {
            None
        }
    }

    pub fn allocate_def_id(&mut self) -> DefId {
        let id = DefId(self.next_def_id);
        self.next_def_id += 1;
        id
    }

    pub fn allocate_crate_id(&mut self) -> CrateId {
        CrateId(self.allocate_def_id())
    }

    /// Look up a method by name on a trait bound at a given DefId.
    pub fn lookup_method(&self, def_id: DefId, name: Symbol) -> Option<&FunctionSignature> {
        if let Some(trait_binding) = self.trait_defs.get(&def_id) {
            trait_binding
                .methods
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, sig)| sig)
        } else {
            None
        }
    }
}
