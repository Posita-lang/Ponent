use crate::ast::Span;

/// A single entry in a call/reference chain (Zig-style `referenced by`).
#[derive(Debug, Clone)]
pub struct ChainEntry {
    pub kind: ChainEntryKind,
    pub span: Span,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainEntryKind {
    /// Direct function call.
    Call,
    /// Inlined call.
    Inlined,
    /// Reference to a declaration.
    Reference,
    /// Defines a symbol.
    Defines,
}

impl ChainEntryKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            ChainEntryKind::Call => "called from here",
            ChainEntryKind::Inlined => "inlined here",
            ChainEntryKind::Reference => "referenced here",
            ChainEntryKind::Defines => "defined here",
        }
    }
}

/// A chain of locations tracing how the compiler reached an error.
/// Inspired by Zig's `referenced by` output.
///
/// Example output:
///   referenced by:
///       main: file.ps:8:12
///       callMain: std/start.zig:698:59
#[derive(Debug, Clone)]
pub struct CallChain {
    entries: Vec<ChainEntry>,
}

impl CallChain {
    pub fn new() -> Self {
        CallChain {
            entries: Vec::new(),
        }
    }

    pub fn push(&mut self, entry: ChainEntry) {
        self.entries.push(entry);
    }

    pub fn push_call(span: Span, message: impl Into<String>) -> Self {
        let mut chain = CallChain::new();
        chain.entries.push(ChainEntry {
            kind: ChainEntryKind::Call,
            span,
            message: message.into(),
        });
        chain
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entries(&self) -> &[ChainEntry] {
        &self.entries
    }

    /// Render the call chain into a multi-line string.
    pub fn render(&self, use_color: bool) -> String {
        if self.entries.is_empty() {
            return String::new();
        }

        let mut out = String::new();
        let header = "referenced by:";
        if use_color {
            out.push_str(&format!("\x1b[36m{}\x1b[0m\n", header));
        } else {
            out.push_str(&format!("{}\n", header));
        }

        for entry in &self.entries {
            let label = entry.kind.as_str();
            if use_color {
                out.push_str(&format!(
                    "    \x1b[34m{}\x1b[0m: {}\n",
                    entry.message, label
                ));
            } else {
                out.push_str(&format!(
                    "    {}: {}\n",
                    entry.message, label
                ));
            }
        }

        out
    }
}

impl Default for CallChain {
    fn default() -> Self {
        Self::new()
    }
}
