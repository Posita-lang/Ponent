/// A set of colour/style values used for rendering diagnostic output.
///
/// Inspired by the `Stylesheet` in `annotate-snippets-rs`.  Each emitter
/// (HTML, terminal, etc.) can use the fields it cares about; unused fields
/// are simply ignored.
#[derive(Debug, Clone)]
pub struct Stylesheet {
    // ── Foreground / background ──
    pub bg: &'static str,
    pub fg: &'static str,
    pub diag_bg: &'static str,
    pub border: &'static str,

    // ── Diagnostic level accents ──
    pub error_border: &'static str,
    pub error_bg: &'static str,
    pub warning_border: &'static str,
    pub warning_bg: &'static str,
    pub help_border: &'static str,
    pub help_bg: &'static str,

    // ── Semantic colours ──
    pub accent: &'static str,
    pub muted: &'static str,
    pub source_fg: &'static str,
    pub suggestion: &'static str,
    pub help_text: &'static str,
    pub highlight_bg: &'static str,
    pub underline: &'static str,
    pub explain_fg: &'static str,
    pub toggle_bg: &'static str,
}

impl Stylesheet {
    /// Dark theme (Catppuccin Mocha) — the default.
    pub const fn dark() -> Self {
        Stylesheet {
            bg: "#1e1e2e",
            fg: "#cdd6f4",
            diag_bg: "#181825",
            border: "#45475a",
            error_border: "#f38ba8",
            error_bg: "#2a1e1e",
            warning_border: "#f9e2af",
            warning_bg: "#2a2a1e",
            help_border: "#94e2d5",
            help_bg: "#1e2a2a",
            accent: "#89b4fa",
            muted: "#6c7086",
            source_fg: "#a6adc8",
            suggestion: "#a6e3a1",
            help_text: "#94e2d5",
            highlight_bg: "#f38ba833",
            underline: "#f38ba8",
            explain_fg: "#bac2de",
            toggle_bg: "#45475a",
        }
    }

    /// Light theme (clean white / Material Design inspired).
    pub const fn light() -> Self {
        Stylesheet {
            bg: "#ffffff",
            fg: "#1e1e2e",
            diag_bg: "#f5f5f5",
            border: "#d0d0d0",
            error_border: "#d32f2f",
            error_bg: "#fce4e4",
            warning_border: "#f57c00",
            warning_bg: "#fff3e0",
            help_border: "#00796b",
            help_bg: "#e0f2f1",
            accent: "#1565c0",
            muted: "#757575",
            source_fg: "#424242",
            suggestion: "#2e7d32",
            help_text: "#00796b",
            highlight_bg: "#ffcdd266",
            underline: "#d32f2f",
            explain_fg: "#616161",
            toggle_bg: "#e0e0e0",
        }
    }

    /// Render the stylesheet as a CSS string for the HTML emitter.
    pub fn to_html_css(&self) -> String {
        format!(
            r#":root {{
            --bg: {bg};
            --fg: {fg};
            --diag-bg: {diag_bg};
            --border: {border};
            --error-border: {error_border};
            --error-bg: {error_bg};
            --warning-border: {warning_border};
            --warning-bg: {warning_bg};
            --help-border: {help_border};
            --help-bg: {help_bg};
            --accent: {accent};
            --muted: {muted};
            --source-fg: {source_fg};
            --suggestion: {suggestion};
            --help-text: {help_text};
            --highlight-bg: {highlight_bg};
            --underline: {underline};
            --explain-fg: {explain_fg};
            --toggle-bg: {toggle_bg};
        }}
        /* ── Light theme (activated by adding .light to <body>) ── */
        body.light {{
            --bg: {light_bg};
            --fg: {light_fg};
            --diag-bg: {light_diag_bg};
            --border: {light_border};
            --error-border: {light_error_border};
            --error-bg: {light_error_bg};
            --warning-border: {light_warning_border};
            --warning-bg: {light_warning_bg};
            --help-border: {light_help_border};
            --help-bg: {light_help_bg};
            --accent: {light_accent};
            --muted: {light_muted};
            --source-fg: {light_source_fg};
            --suggestion: {light_suggestion};
            --help-text: {light_help_text};
            --highlight-bg: {light_highlight_bg};
            --underline: {light_underline};
            --explain-fg: {light_explain_fg};
            --toggle-bg: {light_toggle_bg};
        }}"#,
            bg = self.bg,
            fg = self.fg,
            diag_bg = self.diag_bg,
            border = self.border,
            error_border = self.error_border,
            error_bg = self.error_bg,
            warning_border = self.warning_border,
            warning_bg = self.warning_bg,
            help_border = self.help_border,
            help_bg = self.help_bg,
            accent = self.accent,
            muted = self.muted,
            source_fg = self.source_fg,
            suggestion = self.suggestion,
            help_text = self.help_text,
            highlight_bg = self.highlight_bg,
            underline = self.underline,
            explain_fg = self.explain_fg,
            toggle_bg = self.toggle_bg,
            light_bg = Self::light().bg,
            light_fg = Self::light().fg,
            light_diag_bg = Self::light().diag_bg,
            light_border = Self::light().border,
            light_error_border = Self::light().error_border,
            light_error_bg = Self::light().error_bg,
            light_warning_border = Self::light().warning_border,
            light_warning_bg = Self::light().warning_bg,
            light_help_border = Self::light().help_border,
            light_help_bg = Self::light().help_bg,
            light_accent = Self::light().accent,
            light_muted = Self::light().muted,
            light_source_fg = Self::light().source_fg,
            light_suggestion = Self::light().suggestion,
            light_help_text = Self::light().help_text,
            light_highlight_bg = Self::light().highlight_bg,
            light_underline = Self::light().underline,
            light_explain_fg = Self::light().explain_fg,
            light_toggle_bg = Self::light().toggle_bg,
        )
    }
}
