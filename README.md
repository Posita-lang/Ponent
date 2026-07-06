# Ponent

ponent: official Posita compiler, translating source to native code via Cranelift with an optional strict mode for SMT‑based contract verification. The compiler frontend is well under development — a fully tested lexer, a recursive‑descent parser (5300+ lines, 100+ tests), a rich type system (3000+ lines), a type checker with trait resolution and region checking (2300+ lines), a name resolver, and an SMT solver bridge (Z3) for contract verification are all in place.

