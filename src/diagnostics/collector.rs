use super::{Diagnostic, EmissionGuarantee};

/// Collects multiple diagnostics and provides utilities for error aggregation.
/// Analogue to rustc's `DiagnosticBuilder` / `Handler` pattern.
#[derive(Debug, Clone)]
pub struct DiagnosticCollector {
    diagnostics: Vec<Diagnostic>,
}

impl DiagnosticCollector {
    pub fn new() -> Self {
        DiagnosticCollector {
            diagnostics: Vec::new(),
        }
    }

    /// Push a diagnostic into the collector and return an `EmissionGuarantee`
    /// proving the diagnostic was recorded (inspired by rustc's `ErrorGuaranteed`).
    pub fn push(&mut self, diag: Diagnostic) -> EmissionGuarantee {
        self.diagnostics.push(diag);
        EmissionGuarantee::emitted()
    }

    pub fn extend(&mut self, diags: Vec<Diagnostic>) {
        self.diagnostics.extend(diags);
    }

    pub fn is_empty(&self) -> bool {
        self.diagnostics.is_empty()
    }

    pub fn len(&self) -> usize {
        self.diagnostics.len()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Diagnostic> {
        self.diagnostics.iter()
    }

    pub fn into_inner(self) -> Vec<Diagnostic> {
        self.diagnostics
    }

    pub fn has_errors(&self) -> bool {
        self.diagnostics.iter().any(|d| d.is_error())
    }

    pub fn clear(&mut self) {
        self.diagnostics.clear();
    }

    pub fn emit(&self, emitter: &mut dyn super::emitter::DiagnosticEmitter) {
        emitter.emit_all(&self.diagnostics);
    }

    /// Drain all diagnostics and return them as a Result.
    /// Returns Ok(()) if no errors, Err(self) if any errors present.
    ///
    /// # Errors
    ///
    /// Returns `Err(self)` if any errors were collected.  The caller can
    /// inspect the diagnostics via `self.errors()` or `self.warnings()`.
    #[must_use]
    pub fn check(self) -> Result<(), Self> {
        if self.has_errors() { Err(self) } else { Ok(()) }
    }

    /// Returns all error-level diagnostics as a slice.
    pub fn errors(&self) -> Vec<&Diagnostic> {
        self.diagnostics.iter().filter(|d| d.is_error()).collect()
    }

    /// Returns all warning-level diagnostics as a slice.
    pub fn warnings(&self) -> Vec<&Diagnostic> {
        self.diagnostics.iter().filter(|d| d.is_warning()).collect()
    }

    /// Iterate over diagnostics, yielding them with 1-based index for display.
    pub fn enumerate(&self) -> impl Iterator<Item = (usize, &Diagnostic)> {
        self.diagnostics.iter().enumerate().map(|(i, d)| (i + 1, d))
    }

    /// Total count of error diagnostics.
    pub fn error_count(&self) -> usize {
        self.diagnostics.iter().filter(|d| d.is_error()).count()
    }

    /// Total count of warning diagnostics.
    pub fn warning_count(&self) -> usize {
        self.diagnostics.iter().filter(|d| d.is_warning()).count()
    }
}

impl Default for DiagnosticCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl From<Vec<Diagnostic>> for DiagnosticCollector {
    fn from(diags: Vec<Diagnostic>) -> Self {
        DiagnosticCollector { diagnostics: diags }
    }
}

impl IntoIterator for DiagnosticCollector {
    type Item = Diagnostic;
    type IntoIter = std::vec::IntoIter<Diagnostic>;

    fn into_iter(self) -> Self::IntoIter {
        self.diagnostics.into_iter()
    }
}
