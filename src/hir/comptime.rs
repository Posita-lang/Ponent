use crate::diagnostics::Diagnostic;
use crate::hir::checker::TypeChecker;
use crate::hir::hir::HirExpr;
use crate::hir::types::TypeId;

/// The result of evaluating a comptime block:
/// either a concrete value or a type.
#[derive(Debug, Clone)]
pub enum ComptimeValue {
    Type(TypeId),
    Value(HirExpr),
    Error,
}

/// A lazy iterator over comptime-generated values, following the KSP
/// `has_next()`/`next()` pattern (see YenTopKShortestPathsAlg).
///
/// Unlike Rust's standard `Iterator`, this allows the generator to
/// maintain mutable state that the caller can inspect between calls.
/// The sequence is produced on demand — "next" only computes when called.
#[derive(Debug, Clone)]
pub struct SeqGen<T: Clone> {
    /// The buffered candidate values, sorted by priority.
    candidates: Vec<T>,
    /// Maximum number of values to generate (like KSP's `top_k`).
    limit: usize,
    /// Number of values produced so far.
    count: usize,
    /// Generator function: given the last value, produces the next batch.
    /// Returns `None` when no more values can be generated.
    generator: Option<fn(&T) -> Option<Vec<T>>>,
}

impl<T: Clone> SeqGen<T> {
    /// Create a new sequence generator with the first value and a limit.
    pub fn new(first: T, limit: usize, generator: fn(&T) -> Option<Vec<T>>) -> Self {
        SeqGen {
            candidates: vec![first],
            limit,
            count: 0,
            generator: Some(generator),
        }
    }

    /// Check if more values are available (KSP-style `has_next()`).
    /// Returns `true` if `next()` would return `Some`.
    pub fn has_next(&self) -> bool {
        !self.candidates.is_empty() && self.count < self.limit
    }

    /// Produce the next value (KSP-style `next()`).
    /// Returns `None` when no more values can be generated.
    pub fn next(&mut self) -> Option<T> {
        if self.candidates.is_empty() || self.count >= self.limit {
            return None;
        }

        // Take the highest-priority candidate (first in sorted order)
        let result = self.candidates.remove(0);
        self.count += 1;

        // If we haven't reached the limit, generate the next batch
        // from the just-produced value
        if self.count < self.limit {
            if let Some(gen) = self.generator {
                if let Some(new_candidates) = gen(&result) {
                    for c in new_candidates {
                        // Insert maintaining priority order (ascending by default)
                        let pos = self.candidates.iter().position(|x| {
                            self.priority(x) > self.priority(&c)
                        });
                        match pos {
                            Some(p) => self.candidates.insert(p, c),
                            None => self.candidates.push(c),
                        }
                    }
                }
            }
        }

        Some(result)
    }

    /// Priority function — override for custom ordering.
    /// Lower values come first (like Dijkstra's `WeightLess`).
    fn priority(&self, _val: &T) -> usize {
        0 // Default: insertion order
    }
}

/// A comptime iterator that yields types from a type factory function.
/// Mirrors the KSP `YenTopKShortestPathsAlg` usage pattern:
/// ```
/// let mut gen = TypeSeqGen::new(initial_type, 5, next_type_factory);
/// while gen.has_next() {
///     let t = gen.next().unwrap();
///     // use t at compile time
/// }
/// ```
#[derive(Debug, Clone)]
pub struct TypeSeqGen {
    inner: SeqGen<TypeId>,
}

impl TypeSeqGen {
    pub fn new(
        first: TypeId,
        limit: usize,
        generator: fn(&TypeId) -> Option<Vec<TypeId>>,
    ) -> Self {
        TypeSeqGen {
            inner: SeqGen::new(first, limit, generator),
        }
    }

    pub fn has_next(&self) -> bool {
        self.inner.has_next()
    }

    pub fn next(&mut self) -> Option<TypeId> {
        self.inner.next()
    }
}

/// A comptime iterator that yields expressions from an expression factory.
#[derive(Debug, Clone)]
pub struct ExprSeqGen {
    inner: SeqGen<HirExpr>,
}

impl ExprSeqGen {
    pub fn new(
        first: HirExpr,
        limit: usize,
        generator: fn(&HirExpr) -> Option<Vec<HirExpr>>,
    ) -> Self {
        ExprSeqGen {
            inner: SeqGen::new(first, limit, generator),
        }
    }

    pub fn has_next(&self) -> bool {
        self.inner.has_next()
    }

    pub fn next(&mut self) -> Option<HirExpr> {
        self.inner.next()
    }
}

/// Evaluation context for comptime blocks.
/// Comptime blocks are evaluated during type-checking and must be
/// side-effect free (no @io, no file access, no external calls).
pub struct ComptimeEvalContext<'a> {
    /// Reference to the parent type checker.
    pub checker: &'a mut TypeChecker<'a>,
    /// Maximum evaluation steps before bailing out.
    pub step_limit: usize,
    /// Current step count.
    pub steps: usize,
}

impl<'a> ComptimeEvalContext<'a> {
    /// Create a new comptime evaluation context.
    pub fn new(checker: &'a mut TypeChecker<'a>) -> Self {
        ComptimeEvalContext {
            checker,
            step_limit: 1000,
            steps: 0,
        }
    }

    /// Check whether the given expression is allowed in comptime context.
    /// Comptime blocks cannot contain I/O, file access, or other side effects.
    pub fn check_comptime_allowed(expr: &HirExpr) -> bool {
        match expr {
            HirExpr::Literal(..) => true,
            HirExpr::Ident(..) => true,
            HirExpr::BinaryOp { .. } => true,
            HirExpr::UnaryOp { .. } => true,
            HirExpr::Tuple(..) => true,
            HirExpr::Array(..) => true,
            HirExpr::Call { .. } => {
                // Function calls in comptime are allowed only if the function
                // is marked `@pure` or is itself a comptime function.
                // For now, conservatively allow all calls.
                true
            }
            // Side-effectful operations are not comptime-safe:
            HirExpr::FieldAccess { .. } => true,
            _ => {
                // I/O, file access, and other impure operations are disallowed.
                // This is a conservative starting point.
                false
            }
        }
    }

    /// Evaluate a comptime expression to a value.
    /// Returns `None` if the expression cannot be evaluated at comptime.
    pub fn eval_expr(&mut self, _expr: &HirExpr) -> Option<ComptimeValue> {
        // Skeleton: actual evaluation will be implemented in future iterations.
        // For now, this returns None to signal "deferred to runtime".
        None
    }
}
