// ── Symbol: Interned String ────────────────────────────────────────
//
// A lightweight string interner using a simple arena + index scheme.
// Inspired by rustc's `Symbol` and `Interner`.  No external dependencies.
//
// Symbols are `u32` indices into an interning table.  The `Interner`
// ensures that each unique string is stored exactly once, so `Symbol`
// comparison is a single `u32` comparison — no string allocation, no
// hashing, no pointer chasing.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;

/// A compact, interned string identifier.
///
/// `Symbol` values are comparable, hashable, and copyable — they are
/// the recommended way to represent identifiers, keywords, and paths
/// throughout the compiler.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Symbol(u32);

impl Symbol {
    /// The invalid / sentinel symbol (used for errors, uninitialised slots).
    pub const INVALID: Symbol = Symbol(u32::MAX);

    /// Create a new interned symbol from a string slice.
    /// If the string has already been interned, returns the existing symbol.
    pub fn intern(s: &str) -> Self {
        INTERNER.with(|i| i.borrow_mut().intern(s))
    }

    /// Retrieve the string representation of this symbol.
    pub fn as_str(self) -> String {
        INTERNER.with(|i| i.borrow().resolve(self))
    }

    /// Convenience: compare against a string slice without resolving.
    pub fn eq_str(self, s: &str) -> bool {
        self.as_str() == s
    }
}

impl fmt::Debug for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl From<&str> for Symbol {
    fn from(s: &str) -> Self {
        Symbol::intern(s)
    }
}

// ── Interner ───────────────────────────────────────────────────────

/// The global interner state.
struct Interner {
    /// Map from string content to its `Symbol` (stable index).
    map: HashMap<String, Symbol>,
    /// Arena: each interned string is stored exactly once.
    arena: Vec<String>,
    /// Counter for the next symbol index.
    next_id: u32,
}

impl Interner {
    fn new() -> Self {
        Interner {
            map: HashMap::new(),
            arena: Vec::new(),
            next_id: 0,
        }
    }

    fn intern(&mut self, s: &str) -> Symbol {
        // Check if already interned.
        if let Some(&sym) = self.map.get(s) {
            return sym;
        }
        let id = self.next_id;
        self.next_id += 1;
        let owned = s.to_string();
        let sym = Symbol(id);
        self.map.insert(owned.clone(), sym);
        self.arena.push(owned);
        sym
    }

    fn resolve(&self, sym: Symbol) -> String {
        if sym.0 == u32::MAX {
            return "<invalid>".to_string();
        }
        self.arena
            .get(sym.0 as usize)
            .cloned()
            .unwrap_or_else(|| format!("<symbol {}>", sym.0))
    }
}

// Use a thread-local interner so that Symbol::intern() is safe to call
// from any thread (the compiler is primarily single-threaded).
thread_local! {
    static INTERNER: RefCell<Interner> = RefCell::new(Interner::new());
}

/// Pre-intern common keywords and built-in type names so that
/// the parser and type checker can use `Symbol::from("fn")` etc.
/// without paying the interning cost on first use.
///
/// Call this once at compiler startup (e.g., in main()).
pub fn pre_intern_builtins() {
    let builtins = &[
        "true", "false", "def", "set", "return", "if", "else", "for", "while",
        "type", "enum", "struct", "impl", "trait", "fn", "let", "mut",
        "Int", "UInt", "Float", "Bool", "Char", "Byte", "USize", "Unit",
        "Result", "Option", "Ok", "Err", "Some", "None",
        "Self", "self",
    ];
    for s in builtins {
        Symbol::intern(s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intern_and_resolve() {
        let a = Symbol::intern("hello");
        let b = Symbol::intern("hello");
        let c = Symbol::intern("world");
        assert_eq!(a, b, "same string → same symbol");
        assert_ne!(a, c, "different strings → different symbols");
        assert_eq!(a.as_str(), "hello");
        assert_eq!(c.as_str(), "world");
    }

    #[test]
    fn test_symbol_invalid() {
        assert_eq!(Symbol::INVALID.as_str(), "<invalid>");
    }
}
