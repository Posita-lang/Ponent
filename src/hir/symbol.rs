use crate::ast::AssociatedType;
use crate::ast::Attribute;
use crate::ast::Contract;
use crate::ast::Span;
use crate::ast::TraitMethod;
use crate::ast::TypeParam;
use crate::hir::types::{DefId, TypeError, TypeId};
use std::collections::HashMap;
use std::collections::hash_map::Entry;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BindingKind {
    Variable,
    Function,
    Type,
    Trait,
    Impl,
    Module,
}

#[derive(Debug, Clone)]
pub struct VariableBinding {
    pub ty: TypeId,
    pub mutable: bool,
    pub span: Span,
    pub def_id: DefId,
    pub is_ghost: bool,
    pub is_comptime: bool,
}

#[derive(Debug, Clone)]
pub struct FunctionSignature {
    pub params: Vec<(String, TypeId)>,
    pub ret: TypeId,
    pub type_params: Vec<TypeParam>,
    pub variadic: bool,
}

#[derive(Debug, Clone)]
pub struct FunctionBinding {
    pub signature: FunctionSignature,
    pub def_id: DefId,
    pub is_comptime: bool,
    pub is_async: bool,
    pub is_pure: bool,
    pub is_trusted: bool,
    pub contracts: Vec<Contract>,
    pub attributes: Vec<Attribute>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeKind {
    Struct,
    Enum,
    Alias,
    Opaque,
}

#[derive(Debug, Clone)]
pub struct TypeBinding {
    pub def_id: DefId,
    pub params: Vec<TypeParam>,
    pub kind: TypeKind,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct TraitBinding {
    pub def_id: DefId,
    pub params: Vec<TypeParam>,
    pub methods: Vec<TraitMethod>,
    pub associated_types: Vec<AssociatedType>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct ImplBinding {
    pub def_id: DefId,
    pub trait_path: Option<Vec<String>>,
    pub for_type: TypeId,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct ModuleBinding {
    pub name: String,
    pub def_id: DefId,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum Binding {
    Variable(VariableBinding),
    Function(FunctionBinding),
    Type(TypeBinding),
    Trait(TraitBinding),
    Impl(ImplBinding),
    Module(ModuleBinding),
}

#[derive(Debug, Clone)]
pub struct Scope {
    pub parent: Option<usize>,
    pub variables: HashMap<String, VariableBinding>,
    pub functions: HashMap<String, FunctionBinding>,
    pub types: HashMap<String, TypeBinding>,
    pub traits: HashMap<String, TraitBinding>,
    pub impls: Vec<ImplBinding>,
    pub modules: HashMap<String, ModuleBinding>,
}

impl Scope {
    pub fn new(parent: Option<usize>) -> Self {
        Scope {
            parent,
            variables: HashMap::new(),
            functions: HashMap::new(),
            types: HashMap::new(),
            traits: HashMap::new(),
            impls: Vec::new(),
            modules: HashMap::new(),
        }
    }
}

pub struct SymbolTable {
    pub scopes: Vec<Scope>,
    pub current_scope: usize,
    pub def_counter: usize,
    pub type_id_counter: usize,
}

impl SymbolTable {
    pub fn new() -> Self {
        let root = Scope::new(None);
        SymbolTable {
            scopes: vec![root],
            current_scope: 0,
            def_counter: 0,
            type_id_counter: 0,
        }
    }

    pub fn push_scope(&mut self) -> usize {
        let parent = self.current_scope;
        let scope = Scope::new(Some(parent));
        let id = self.scopes.len();
        self.scopes.push(scope);
        self.current_scope = id;
        id
    }

    pub fn pop_scope(&mut self) {
        if let Some(parent) = self.scopes[self.current_scope].parent {
            self.current_scope = parent;
        }
    }

    pub fn current(&mut self) -> &mut Scope {
        &mut self.scopes[self.current_scope]
    }

    pub fn lookup_variable(&self, name: &str) -> Option<&VariableBinding> {
        let mut scope = self.current_scope;
        while let Some(sc) = self.scopes.get(scope) {
            if let Some(binding) = sc.variables.get(name) {
                return Some(binding);
            }
            if let Some(parent) = sc.parent {
                scope = parent;
            } else {
                break;
            }
        }
        None
    }

    pub fn lookup_function(&self, name: &str) -> Option<&FunctionBinding> {
        let mut scope = self.current_scope;
        while let Some(sc) = self.scopes.get(scope) {
            if let Some(binding) = sc.functions.get(name) {
                return Some(binding);
            }
            if let Some(parent) = sc.parent {
                scope = parent;
            } else {
                break;
            }
        }
        None
    }

    pub fn lookup_type(&self, name: &str) -> Option<&TypeBinding> {
        let mut scope = self.current_scope;
        while let Some(sc) = self.scopes.get(scope) {
            if let Some(binding) = sc.types.get(name) {
                return Some(binding);
            }
            if let Some(parent) = sc.parent {
                scope = parent;
            } else {
                break;
            }
        }
        None
    }

    pub fn lookup_trait(&self, name: &str) -> Option<&TraitBinding> {
        let mut scope = self.current_scope;
        while let Some(sc) = self.scopes.get(scope) {
            if let Some(binding) = sc.traits.get(name) {
                return Some(binding);
            }
            if let Some(parent) = sc.parent {
                scope = parent;
            } else {
                break;
            }
        }
        None
    }

    pub fn lookup_module(&self, name: &str) -> Option<&ModuleBinding> {
        let mut scope = self.current_scope;
        while let Some(sc) = self.scopes.get(scope) {
            if let Some(binding) = sc.modules.get(name) {
                return Some(binding);
            }
            if let Some(parent) = sc.parent {
                scope = parent;
            } else {
                break;
            }
        }
        None
    }

    pub fn insert_variable(
        &mut self,
        name: &str,
        binding: VariableBinding,
    ) -> Result<(), TypeError> {
        match self.current().variables.entry(name.to_string()) {
            Entry::Occupied(entry) => Err(TypeError::DuplicateDefinition {
                name: name.to_string(),
                span: binding.span,
                previous: entry.get().span,
            }),
            Entry::Vacant(entry) => {
                entry.insert(binding);
                Ok(())
            }
        }
    }

    pub fn insert_function(
        &mut self,
        name: &str,
        binding: FunctionBinding,
    ) -> Result<(), TypeError> {
        match self.current().functions.entry(name.to_string()) {
            Entry::Occupied(entry) => Err(TypeError::DuplicateDefinition {
                name: name.to_string(),
                span: binding.span,
                previous: entry.get().span,
            }),
            Entry::Vacant(entry) => {
                entry.insert(binding);
                Ok(())
            }
        }
    }

    pub fn insert_type(&mut self, name: &str, binding: TypeBinding) -> Result<(), TypeError> {
        match self.current().types.entry(name.to_string()) {
            Entry::Occupied(entry) => Err(TypeError::DuplicateDefinition {
                name: name.to_string(),
                span: binding.span,
                previous: entry.get().span,
            }),
            Entry::Vacant(entry) => {
                entry.insert(binding);
                Ok(())
            }
        }
    }

    pub fn insert_trait(&mut self, name: &str, binding: TraitBinding) -> Result<(), TypeError> {
        match self.current().traits.entry(name.to_string()) {
            Entry::Occupied(entry) => Err(TypeError::DuplicateDefinition {
                name: name.to_string(),
                span: binding.span,
                previous: entry.get().span,
            }),
            Entry::Vacant(entry) => {
                entry.insert(binding);
                Ok(())
            }
        }
    }

    pub fn insert_impl(&mut self, binding: ImplBinding) {
        self.current().impls.push(binding);
    }

    pub fn insert_module(&mut self, name: &str, binding: ModuleBinding) -> Result<(), TypeError> {
        match self.current().modules.entry(name.to_string()) {
            Entry::Occupied(entry) => Err(TypeError::DuplicateDefinition {
                name: name.to_string(),
                span: binding.span,
                previous: entry.get().span,
            }),
            Entry::Vacant(entry) => {
                entry.insert(binding);
                Ok(())
            }
        }
    }

    pub fn resolve_path(&self, path: &[String]) -> Option<DefId> {
        if path.is_empty() {
            return None;
        }
        let mut scope = self.current_scope;
        let mut current = Vec::new();
        let mut found = None;
        for (i, seg) in path.iter().enumerate() {
            if i == 0 {
                if let Some(module) = self.lookup_module(seg) {
                    found = Some(module.def_id);
                    continue;
                }
                if let Some(ty) = self.lookup_type(seg) {
                    if path.len() == 1 {
                        return Some(ty.def_id);
                    } else {
                        return None;
                    }
                }
                if let Some(trait_) = self.lookup_trait(seg) {
                    if path.len() == 1 {
                        return Some(trait_.def_id);
                    } else {
                        return None;
                    }
                }
                return None;
            } else {
                if let Some(def_id) = found {
                    return Some(def_id);
                }
                return None;
            }
        }
        found
    }

    pub fn next_def_id(&mut self) -> DefId {
        let id = self.def_counter;
        self.def_counter += 1;
        DefId(id)
    }

    pub fn current_scope_id(&self) -> usize {
        self.current_scope
    }

    pub fn get_scope(&self, id: usize) -> Option<&Scope> {
        self.scopes.get(id)
    }

    pub fn get_scope_mut(&mut self, id: usize) -> Option<&mut Scope> {
        self.scopes.get_mut(id)
    }
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self::new()
    }
}
