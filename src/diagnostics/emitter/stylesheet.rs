//! Stylesheet generation for the HTML renderer and the explain server.
//!
//! # Compact by construction
//!
//! The generators emit minified CSS directly (`:root{--bg:#1e1e2e;…}`), so there
//! is no runtime minify pass and no second cached copy: a "pretty" variant would
//! only add a string to strip.  That was the earlier design — generate indented
//! text, then post-process it — and it existed only because the layout was
//! written for human eyes first; nothing consumes the un-minified form except
//! tests, which assert declarations rather than layout.
//!
//! The served CSS is a **compile-time constant** (`theme_css!` → `concat!`), so
//! there is no runtime build, no `OnceLock` and no minify pass.  The palette
//! literals appear twice — in the macro list and in [`Stylesheet::dark`] /
//! [`Stylesheet::light`], which the renderers read field by field — and a
//! `const` assert per entry fails the build if the two disagree.
//!
//! The explain server hashes the exact bytes it serves into
//! `Content-Security-Policy` (`style-src`), which is why the served string must
//! be the same constant everywhere rather than built per call.

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
}

impl Default for Stylesheet {
    fn default() -> Self {
        Self::dark()
    }
}

/// Const string equality — used by the palette/CSS drift assert below.
const fn str_eq(a: &str, b: &str) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let (a, b) = (a.as_bytes(), b.as_bytes());
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// Generate the served CSS **at compile time** from one literal list.
///
/// Three outputs from one source:
/// * [`DARK_HTML_CSS`] — `:root` plus the `html.light` override (HTML renderer);
/// * [`LIGHT_HTML_CSS`] — `:root` only (a light page has no toggle copy);
/// * a `const` assert per entry that the literal equals the matching
///   [`Stylesheet`] field, so the palette and the CSS **cannot** drift: a colour
///   edited in one place and not the other is a build error.
///
/// `concat!` requires literals, which is why the values appear both here and in
/// [`Stylesheet::dark`]/[`Stylesheet::light`].  The assert is what makes that
/// duplication safe, and it is why this can be a `&'static str` at all — there is
/// no runtime build, no `OnceLock`, and no minify pass.
macro_rules! theme_css {
    ($($name:literal => $field:ident => $dark:literal => $light:literal),+ $(,)?) => {
        /// Dark theme with the `.light` override — the HTML renderer's stylesheet.
        pub const DARK_HTML_CSS: &str = concat!(
            ":root{color-scheme:dark light;", $($name, ":", $dark, ";",)+ "}",
            "html.light{color-scheme:dark light;", $($name, ":", $light, ";",)+ "}",
        );

        /// Light theme, `:root` only.
        pub const LIGHT_HTML_CSS: &str = concat!(
            ":root{color-scheme:dark light;", $($name, ":", $light, ";",)+ "}",
        );

        /// Every CSS variable this palette defines, in declaration order.
        pub const THEME_VARS: &[&str] = &[$($name),+];

        const _: () = {
            // Completeness AND value agreement, both at compile time.
            //
            // The literals below are built *from the palette*, so a `Stylesheet`
            // field that has no palette line leaves the literal incomplete and
            // is a hard compile error (E0063) — previously such a field was
            // silently unstyled (its `var(--x)` consumers just lost the
            // property) with no build or test signal.  Comparing them against the
            // hand-written constructors then pins the *values* in both
            // directions, so neither home can drift.
            let dark = Stylesheet { $($field: $dark,)+ };
            let light = Stylesheet { $($field: $light,)+ };
            // The comparison must be *between* the two homes, not between the
            // literal and itself: `dark.$field` IS `$dark`, so the earlier form
            // asserted `x == x` and the drift guard this macro exists for did not
            // exist.  `Stylesheet::dark()` / `light()` are the hand-written
            // constructors the renderers read, so pin those against the palette.
            let handwritten_dark = Stylesheet::dark();
            let handwritten_light = Stylesheet::light();
            $(
                assert!(str_eq(dark.$field, handwritten_dark.$field));
                assert!(str_eq(light.$field, handwritten_light.$field));
            )+
        };
    };
}

theme_css!(
    "--bg" => bg => "#1e1e2e" => "#ffffff",
    "--fg" => fg => "#cdd6f4" => "#1e1e2e",
    "--diag-bg" => diag_bg => "#181825" => "#f5f5f5",
    "--border" => border => "#45475a" => "#d0d0d0",
    "--error-border" => error_border => "#f38ba8" => "#d32f2f",
    "--error-bg" => error_bg => "#2a1e1e" => "#fce4e4",
    "--warning-border" => warning_border => "#f9e2af" => "#f57c00",
    "--warning-bg" => warning_bg => "#2a2a1e" => "#fff3e0",
    "--help-border" => help_border => "#94e2d5" => "#00796b",
    "--help-bg" => help_bg => "#1e2a2a" => "#e0f2f1",
    "--accent" => accent => "#89b4fa" => "#1565c0",
    "--muted" => muted => "#6c7086" => "#757575",
    "--source-fg" => source_fg => "#a6adc8" => "#424242",
    "--suggestion" => suggestion => "#a6e3a1" => "#2e7d32",
    "--help-text" => help_text => "#94e2d5" => "#00796b",
    "--highlight-bg" => highlight_bg => "#f38ba833" => "#ffcdd266",
    "--underline" => underline => "#f38ba8" => "#d32f2f",
    "--explain-fg" => explain_fg => "#bac2de" => "#616161",
    "--toggle-bg" => toggle_bg => "#45475a" => "#e0e0e0",
);

/// The dark-theme stylesheet, as a compile-time constant.
pub fn dark_css() -> &'static str {
    DARK_HTML_CSS
}

/// The light-theme stylesheet (`:root` only), as a compile-time constant.
///
/// No in-tree consumer calls this: the renderers and the explain server serve
/// [`dark_css`], whose `html.light` block carries the light overrides for a page
/// that opts in.  `LIGHT_HTML_CSS` exists for a standalone light-only page
/// (external/static publishing), and `theme_vars`-style completeness tests pin it
/// against the palette so it cannot drift while unused.
pub fn light_css() -> &'static str {
    LIGHT_HTML_CSS
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;

    /// The completeness half of the palette guarantee.
    ///
    /// `theme_css!` now *builds* both `Stylesheet` literals from the palette, so
    /// a palette line dropped from the invocation is a compile error (missing
    /// field) rather than a silently unstyled CSS variable.  This test closes the
    /// other direction: every declared variable must actually reach both strings
    /// — exactly twice in `DARK_HTML_CSS` (`:root` plus the `html.light`
    /// override) and once in `LIGHT_HTML_CSS`.
    #[test]
    fn every_theme_var_reaches_both_stylesheets_exactly_once_per_scope() {
        assert!(!THEME_VARS.is_empty());
        assert_eq!(
            THEME_VARS.len(),
            19,
            "palette size changed — update the pin"
        );
        for var in THEME_VARS {
            let needle = format!("{var}:");
            assert_eq!(
                DARK_HTML_CSS.matches(&needle).count(),
                2,
                "{var} must appear in `:root` and in `html.light`"
            );
            assert_eq!(
                LIGHT_HTML_CSS.matches(&needle).count(),
                1,
                "{var} must appear once in the light `:root`"
            );
        }
        // A page opting into `html.light` must get byte-identical declarations to
        // a light-only page, or the two light paths diverge.
        let root = LIGHT_HTML_CSS
            .strip_prefix(":root{")
            .and_then(|rest| rest.strip_suffix('}'))
            .expect("light stylesheet shape changed");
        let override_body = DARK_HTML_CSS
            .split_once("html.light{")
            .and_then(|(_, rest)| rest.strip_suffix('}'))
            .expect("dark stylesheet shape changed");
        assert_eq!(root, override_body);
    }

    /// The generators emit compact CSS directly: no newlines, no indentation,
    /// no comments.  This is what replaced the runtime `minify_css` pass — if a
    /// pretty-printed block creeps back in, the served bytes silently grow and
    /// the CSP hash changes with them.
    #[test]
    fn generated_css_is_compact_by_construction() {
        let dark = dark_css();
        assert!(!dark.contains('\n'), "no layout to strip: {dark:?}");
        assert!(!dark.contains("/*"), "no comments in served CSS");
        assert!(dark.starts_with(":root{"), "{dark:?}");
        assert!(dark.contains("}html.light{"), "both themes in one string");
        assert!(dark.contains("--error-border:"), "{dark:?}");
        assert!(dark.ends_with('}'), "{dark:?}");

        let light = light_css();
        assert!(light.starts_with(":root{"), "{light:?}");
        assert!(
            !light.contains("html.light"),
            "root-only theme has no toggle copy"
        );
        assert!(light.ends_with('}'), "{light:?}");

        // Compactness is not bought with truncation: every palette declaration
        // survives, in both theme blocks.  An exact count (rather than the old
        // `% 2 == 0`, which any even number satisfied) makes a dropped variable
        // a failure; the `color-scheme` declaration each `:root` carries is the
        // only other `;`.
        assert_eq!(
            dark.matches(';').count(),
            2 * THEME_VARS.len() + 2,
            "declaration count should match the palette exactly"
        );
        assert_eq!(light.matches(';').count(), THEME_VARS.len() + 1);
    }
}
