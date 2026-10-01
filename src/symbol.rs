// ── Symbol: Interned String ────────────────────────────────────────
//
// A lightweight string interner using a simple arena + index scheme.
// Inspired by rustc's `Symbol` and `Interner`.  No external dependencies.
//
// Symbols are `u32` indices into an interning table.  The `Interner`
// ensures that each unique string is stored exactly once, so `Symbol`
// comparison is a single `u32` comparison — no string allocation, no
// hashing, no pointer chasing.

use std::collections::HashMap;
use std::fmt;

/// A compact, interned string identifier.
///
/// `Symbol` values are comparable, hashable, and copyable — they are
/// the recommended way to represent identifiers, keywords, and paths
/// throughout the compiler.
///
/// ── Sibling ───────────────────────────────────────────────────
/// The historical fork in `src/hir/symbol_table.rs` was the same
/// Symbol/Interner types.  That file has been deleted and all
/// references consolidated into this module.  The implementation
/// here includes `resolve_ref` which the fork lacked.
/// ──────────────────────────────────────────────────────────────
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Symbol(u32);

impl Symbol {
    /// The invalid / sentinel symbol (used for errors, uninitialised slots).
    pub const INVALID: Symbol = Symbol(u32::MAX);

    /// The underlying interner index (used by the Polonius fact extractor
    /// to encode variables as plain integers).
    pub fn to_u32(self) -> u32 {
        self.0
    }

    /// Create a new interned symbol from a string slice.
    /// If the string has already been interned, returns the existing symbol.
    pub fn intern(s: &str) -> Self {
        let guard = INTERNER.read();
        if let Some(&sym) = guard.map.get(s) {
            return sym;
        }
        drop(guard);
        INTERNER.write().intern(s)
    }

    /// Retrieve the string representation of this symbol.
    ///
    /// Returns a `&'static str` view into the process-global interner arena.
    /// The arena is append-only and leaked for the whole process lifetime, so
    /// the borrow is valid without holding the interner read lock — callers
    /// never need to `.to_string()` / `.clone()` the result just to keep it
    /// alive, and `Cow::Owned(name.as_str())` becomes `Cow::Borrowed(name.as_str())`.
    pub fn as_str(self) -> &'static str {
        INTERNER.read().resolve_ref(self)
    }

    /// Compare against a string slice without allocating.
    /// Internally resolves to `&str` inside the interner lock, avoiding
    /// the `String` allocation that `as_str()` would produce.
    pub fn eq_str(self, s: &str) -> bool {
        INTERNER.read().resolve_ref(self) == s
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
///
/// Arena entries are leaked (`&'static str`) so that a resolved name stays
/// valid after the lock is released — the arena is append-only and lives for
/// the whole process anyway.
struct Interner {
    /// Map from string content to its `Symbol` (stable index).
    map: HashMap<String, Symbol>,
    /// Arena: each interned string is stored exactly once.  Index == id.
    arena: Vec<&'static str>,
}

impl Interner {
    fn new() -> Self {
        Interner {
            map: HashMap::new(),
            arena: Vec::new(),
        }
    }

    fn intern(&mut self, s: &str) -> Symbol {
        // Check if already interned (a writer may have won the race between
        // our read-lock probe and this write lock).
        if let Some(&sym) = self.map.get(s) {
            return sym;
        }
        let sym = Symbol(self.arena.len() as u32);
        self.map.insert(s.to_string(), sym);
        self.arena.push(Box::leak(s.to_string().into_boxed_str()));
        sym
    }

    /// Resolve a symbol to a `&str` reference into the arena (no allocation).
    fn resolve_ref(&self, sym: Symbol) -> &'static str {
        if sym.0 == u32::MAX {
            return "<invalid>";
        }
        self.arena
            .get(sym.0 as usize)
            .copied()
            .unwrap_or("<unknown>")
    }
}

// A PROCESS-global interner, not a thread-local one.
//
// `Symbol` is a bare `u32` index into the arena, so a per-thread interner
// hands every thread a *different* id for the same name.  The type checker
// runs function bodies on a rayon pool; with a thread-local interner a
// worker resolving `Copy` or `Int<32>` gets a fresh empty arena, misses
// every name lookup, and `as_str()` silently returns `<unknown>`.  Symbol
// identity must therefore be process-wide.
//
// Reads (`as_str` / `eq_str`) take the read lock; `intern` probes under a
// read lock and only upgrades on a miss.
static INTERNER: std::sync::LazyLock<parking_lot::RwLock<Interner>> =
    std::sync::LazyLock::new(|| parking_lot::RwLock::new(Interner::new()));

/// Pre-intern common keywords and built-in type names so that
/// the parser and type checker can use `Symbol::from("fn")` etc.
/// without paying the interning cost on first use.
///
/// Call this once at compiler startup (e.g., in main()).
pub fn pre_intern_builtins() {
    let builtins = &[
        "true", "false", "def", "set", "return", "if", "else", "for", "while", "type", "enum",
        "struct", "impl", "trait", "fn", "let", "mut", "Int", "UInt", "Float", "Bool", "Char",
        "Byte", "USize", "Result", "Option", "Ok", "Err", "Some", "None", "Self", "self",
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
