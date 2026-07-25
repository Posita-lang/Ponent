/// SAT/SMT solver family for Posita.
///
/// - **Solvo** (Latin: "I solve") — a minimal DPLL SAT solver for propositional logic.
///   Handles small formulas (tens of variables) efficiently.
///   Future siblings may include:
///   - **Puto** (Latin: "I think") — a CDCL SAT solver with clause learning
///   - **Aptus** (Latin: "suitable") — a theory solver for SMT
pub mod solvo;

pub use solvo::*;
