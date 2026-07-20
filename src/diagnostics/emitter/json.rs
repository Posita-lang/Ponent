use crate::diagnostics::Diagnostic;
use std::fmt::Write;

/// JSON emitter — outputs diagnostics as a JSON array.
/// Format inspired by Austral's `render_error_to_json`.
/// Builds JSON manually without external dependencies.
pub struct JsonEmitter {
    pretty: bool,
}

impl JsonEmitter {
    pub fn new(pretty: bool) -> Self {
        JsonEmitter { pretty }
    }

    pub fn new_pretty() -> Self {
        JsonEmitter { pretty: true }
    }

    pub fn new_compact() -> Self {
        JsonEmitter { pretty: false }
    }
}

impl Default for JsonEmitter {
    fn default() -> Self {
        Self::new_compact()
    }
}

impl JsonEmitter {
    fn escape_json(s: &str) -> String {
        let mut out = String::with_capacity(s.len() + 2);
        out.push('"');
        for ch in s.chars() {
            match ch {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if c.is_control() => {
                    write!(out, "\\u{:04x}", c as u32).unwrap();
                }
                c => out.push(c),
            }
        }
        out.push('"');
        out
    }

    fn indent(level: usize) -> String {
        "  ".repeat(level)
    }

    fn diagnostic_to_json(&self, diag: &Diagnostic, indent_lvl: usize) -> String {
        let i = indent_lvl;
        let mut s = String::from("{\n");

        // level
        s.push_str(&format!(
            "{}  \"level\": {},\n",
            Self::indent(i),
            Self::escape_json(diag.level.as_str())
        ));

        // message
        s.push_str(&format!(
            "{}  \"message\": {},\n",
            Self::indent(i),
            Self::escape_json(&diag.message)
        ));

        // code
        if let Some(ref code) = diag.code {
            s.push_str(&format!(
                "{}  \"code\": {},\n",
                Self::indent(i),
                Self::escape_json(code.code())
            ));
            s.push_str(&format!(
                "{}  \"title\": {},\n",
                Self::indent(i),
                Self::escape_json(code.title())
            ));
            s.push_str(&format!(
                "{}  \"category\": {},\n",
                Self::indent(i),
                Self::escape_json(code.category().as_str())
            ));
        }

        // span
        if let Some(span) = diag.spans.first() {
            s.push_str(&format!(
                "{}  \"span\": {{\"start\": {}, \"end\": {}}},\n",
                Self::indent(i),
                span.start,
                span.end,
            ));
        }

        // labels
        if !diag.labels.is_empty() {
            s.push_str(&format!("{}  \"labels\": [\n", Self::indent(i)));
            for (idx, lbl) in diag.labels.iter().enumerate() {
                s.push_str(&format!(
                    "{}    {{\"start\": {}, \"end\": {}, \"message\": {}, \"kind\": {:?}}}",
                    Self::indent(i),
                    lbl.span.start,
                    lbl.span.end,
                    Self::escape_json(&lbl.message),
                    lbl.kind,
                ));
                if idx < diag.labels.len() - 1 {
                    s.push(',');
                }
                s.push('\n');
            }
            s.push_str(&format!("{}  ],\n", Self::indent(i)));
        }

        // help
        if let Some(ref help) = diag.help {
            s.push_str(&format!(
                "{}  \"help\": {},\n",
                Self::indent(i),
                Self::escape_json(help)
            ));
        }

        // suggestions
        if !diag.suggestions.is_empty() {
            s.push_str(&format!("{}  \"suggestions\": [\n", Self::indent(i)));
            for (idx, sug) in diag.suggestions.iter().enumerate() {
                s.push_str(&format!(
                    "{}    {}",
                    Self::indent(i + 1),
                    Self::escape_json(sug)
                ));
                if idx < diag.suggestions.len() - 1 {
                    s.push(',');
                }
                s.push('\n');
            }
            s.push_str(&format!("{}  ],\n", Self::indent(i)));
        }

        // call chain
        if let Some(ref chain) = diag.call_chain {
            if !chain.is_empty() {
                s.push_str(&format!("{}  \"callChain\": [\n", Self::indent(i)));
                for (idx, entry) in chain.entries().iter().enumerate() {
                    s.push_str(&format!(
                        "{}    {{\"label\": {}, \"kind\": {:?}}}",
                        Self::indent(i),
                        Self::escape_json(&entry.message),
                        entry.kind,
                    ));
                    if idx < chain.entries().len() - 1 {
                        s.push(',');
                    }
                    s.push('\n');
                }
                s.push_str(&format!("{}  ],\n", Self::indent(i)));
            }
        }

        s.push_str(&format!("{}}}", Self::indent(i)));
        s
    }
}

impl super::DiagnosticEmitter for JsonEmitter {
    fn emit(&mut self, diag: &Diagnostic) {
        let json = self.diagnostic_to_json(diag, 0);
        eprintln!("{}", json);
    }

    fn emit_all(&mut self, diags: &[Diagnostic]) {
        if diags.is_empty() {
            eprintln!("[]");
            return;
        }

        let mut s = String::from("[\n");
        for (idx, diag) in diags.iter().enumerate() {
            s.push_str(&self.diagnostic_to_json(diag, 1));
            if idx < diags.len() - 1 {
                s.push(',');
            }
            s.push('\n');
        }
        s.push(']');
        eprintln!("{}", s);
    }
}
