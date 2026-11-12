/// A set of colour/style values used for rendering diagnostic output.
///
/// Inspired by the `Stylesheet` in `annotate-snippets-rs`.  Each emitter
/// (HTML, terminal, etc.) can use the fields it cares about; unused fields
/// are simply ignored.
#[derive(Debug, Clone, Copy)]
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

    /// Render the `--x: value;` variable list for this theme.
    ///
    /// Factored out of [`Self::to_html_css`] so the `:root` and `body.light`
    /// blocks are generated from one list instead of two hand-maintained
    /// copies: a field added to the struct is a compile error in exactly one
    /// place, and the two blocks cannot drift.
    fn css_vars(&self) -> String {
        // Destructure instead of reading `self.*`: adding a field to
        // `Stylesheet` then fails to compile here, so a new theme variable
        // cannot be silently dropped from the emitted CSS.
        let Stylesheet {
            bg,
            fg,
            diag_bg,
            border,
            error_border,
            error_bg,
            warning_border,
            warning_bg,
            help_border,
            help_bg,
            accent,
            muted,
            source_fg,
            suggestion,
            help_text,
            highlight_bg,
            underline,
            explain_fg,
            toggle_bg,
        } = *self;
        // 19 declarations at ~28 chars each; reserve up front so the `write!`
        // chain does not grow the buffer.
        let mut s = String::with_capacity(640);
        let fields: [(&str, &str); 19] = [
            ("--bg", bg),
            ("--fg", fg),
            ("--diag-bg", diag_bg),
            ("--border", border),
            ("--error-border", error_border),
            ("--error-bg", error_bg),
            ("--warning-border", warning_border),
            ("--warning-bg", warning_bg),
            ("--help-border", help_border),
            ("--help-bg", help_bg),
            ("--accent", accent),
            ("--muted", muted),
            ("--source-fg", source_fg),
            ("--suggestion", suggestion),
            ("--help-text", help_text),
            ("--highlight-bg", highlight_bg),
            ("--underline", underline),
            ("--explain-fg", explain_fg),
            ("--toggle-bg", toggle_bg),
        ];
        use std::fmt::Write as _;
        for (name, value) in fields {
            let _ = writeln!(s, "            {name}: {value};");
        }
        s
    }

    /// Render a standalone `:root` block for this theme — no `html.light`
    /// override.  Used when the page has no theme toggle, so the light theme's
    /// `:root` and its redundant `.light` copy are not both emitted.
    fn to_html_css_root_only(&self) -> String {
        let mut out = String::with_capacity(700);
        out.push_str(":root {\n");
        out.push_str(&self.css_vars());
        out.push_str("        }");
        out
    }

    /// Render the stylesheet as a CSS string for the HTML emitter.
    ///
    /// The receiver supplies the `:root` (default) theme; the `.light`
    /// override is always generated from [`Self::light`].  Callers that want
    /// dark-by-default (with a light toggle) must pass a dark theme, e.g.
    /// `Stylesheet::dark().to_html_css()`.
    pub fn to_html_css(&self) -> String {
        let light = Self::light();
        let mut out = String::with_capacity(1320);
        out.push_str(":root {\n");
        out.push_str(&self.css_vars());
        out.push_str(
            "        }\n        /* ── Light theme (activated by adding .light to <html>) ── */\n        html.light {\n",
        );
        out.push_str(&light.css_vars());
        out.push_str("        }");
        out
    }
}

/// The dark-theme CSS, computed once and reused for the process lifetime.
///
/// Both the HTML renderer's `begin` (once per batch) and the explain server
/// (once per request) share this cache, so the `write!` loop inside
/// [`Stylesheet::to_html_css`] runs only on the first call.
pub fn dark_css_cached() -> &'static str {
    static CSS: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    CSS.get_or_init(|| Stylesheet::dark().to_html_css())
}

/// The light-theme CSS, computed once and reused for the process lifetime.
///
/// Symmetric with [`dark_css_cached`], but emits a standalone `:root` block:
/// a light page has no dark/light toggle, so the redundant `html.light` copy
/// the two-theme template would add is pure bytes here.
pub fn light_css_cached() -> &'static str {
    static CSS: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    CSS.get_or_init(|| Stylesheet::light().to_html_css_root_only())
}
