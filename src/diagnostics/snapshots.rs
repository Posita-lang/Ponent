//! Snapshot tests pinning the *rendered shape* of diagnostics.
//!
//! The targeted tests in [`super::glyph`] and [`super::renderers`] pin
//! semantic invariants — an error code is present, a message appears exactly
//! once, a span is escaped — but they leave the *layout* unconstrained:
//! column widths, box-drawing, the connector column, the ordering of
//! children/suggestions/help, and the `1.`/`2.` numbering of related errors
//! can all drift silently.  These snapshots pin the integrated shape.
//!
//! External `.snap` files live in `src/diagnostics/snapshots/`; they are
//! marked `linguist-generated` in `.gitattributes`.  Like every other test
//! module this one is gated off under `--features shuttle`.

use crate::ast::FileId;
use crate::ast::Span;
use crate::diagnostics::chain::{CallChain, ChainEntry, ChainEntryKind};
use crate::diagnostics::glyph::GlyphRenderer;
use crate::diagnostics::render::{RenderCtx, Renderer};
use crate::diagnostics::renderers::{HtmlRenderer, HumanRenderer, JsonRenderer};
use crate::diagnostics::{
    DiagOpts, Diagnostic, ErrCode, RelatedError, StringPart, StringPartStyle, StyledString,
};

/// Source text for the kitchen sink: one statement per line, leading tabs
/// (display-column arithmetic) and one CJK char (wide-cell arithmetic).
const SRC: &str =
    "def foo(x: Bool) -> Int<32> {\n\tset j = x;\n\tset i = j + 1; // 中\n\treturn i;\n}\n";

/// A diagnostic that exercises every layout branch at once:
///
/// * related-error header numbering (`1.` / `2.`),
/// * a multi-line primary span (lines 2–3) with a multi-line label whose
///   message must render exactly once, on the final line,
/// * a secondary (`~`) label and a multi-line (`-`) label that also spans
///   four lines,
/// * a tab and a CJK char (display-column arithmetic),
/// * a child note with a styled (highlighted) message,
/// * a suggestion, a help line, and a call chain.
fn kitchen_sink() -> Diagnostic<'static> {
    let mut chain = CallChain::new();
    chain.push(ChainEntry {
        kind: ChainEntryKind::Call,
        span: Some(Span::new(FileId::DUMMY, 0, 3)),
        message: "main".into(),
    });
    let styled = StyledString::new(vec![
        StringPart {
            content: "expected ".into(),
            style: StringPartStyle::Normal,
        },
        StringPart {
            content: "Int<32>".into(),
            style: StringPartStyle::Highlighted,
        },
    ]);
    Diagnostic::error("return type mismatch")
        .with_code_str("E030")
        .with_span(Span::new(FileId::DUMMY, 43, 68)) // `set i = j + 1` … into `return`
        .with_additional_span(Span::new(FileId::DUMMY, 73, 74)) // `i` in `return i`
        .with_label(Span::new(FileId::DUMMY, 43, 68), "this is Bool")
        .with_secondary_label(Span::new(FileId::DUMMY, 0, 8), "declared to return Int<32>")
        .with_label(Span::new(FileId::DUMMY, 9, 69), "spans four lines")
        .with_related(RelatedError {
            code: Some(ErrCode::new("E019")),
            message: "duplicate definition of `i`".into(),
            span: Some(Span::new(FileId::DUMMY, 43, 56)),
            label: None,
        })
        .with_highlighted_note(styled)
        .with_suggestion("cast with `as`")
        .with_help("change the return type")
        .with_call_chain(chain)
        .with_file_name("example.ps")
        .with_source(SRC)
}

fn ctx(opts: &DiagOpts) -> RenderCtx<'_> {
    RenderCtx::new(opts, None)
}

/// Plain terminal output, including the call chain that only
/// [`HumanRenderer`] appends.
#[test]
fn human_plain_kitchen_sink() {
    let opts = DiagOpts::default();
    let mut r = HumanRenderer::plain();
    let out = r.render(&kitchen_sink(), &ctx(&opts));
    insta::assert_snapshot!("human_plain_kitchen_sink", out);
}

/// Coloured output.  The OSC 8 hyperlink URL embeds a `canonicalize()`d
/// absolute path, so it is redacted; the CSI colour codes are kept but
/// written as the literal two characters `\e` so the `.snap` file stays
/// reviewable plain text and a missing highlight code still shows up as a
/// diff.
#[test]
fn human_colored_kitchen_sink() {
    let opts = DiagOpts::default();
    let mut r = HumanRenderer::colored();
    let out = r.render(&kitchen_sink(), &ctx(&opts));
    insta::with_settings!({filters => vec![
        // OSC 8 open (machine-specific `file://` URL).
        (r"\x1b\]8;;[^\x1b]*\x1b\\", ""),
        // OSC 8 close.
        (r"\x1b\]8;;\x1b\\", ""),
        // Remaining escapes are CSI colour codes; keep them, rendered as `\e`.
        (r"\x1b", r"\e"),
    ]}, {
        insta::assert_snapshot!("human_colored_kitchen_sink", out);
    });
}

/// HTML fragment (not the document framing — `begin`/`end` are covered by
/// the existing `streams_full_document` test).
#[test]
fn html_kitchen_sink() {
    let opts = DiagOpts::default();
    let mut r = HtmlRenderer::new();
    let out = r.render(&kitchen_sink(), &ctx(&opts));
    insta::assert_snapshot!("html_kitchen_sink", out);
}

/// HTML column arithmetic around a wide (CJK) character.  The kitchen sink
/// only puts `中` at end-of-line, which pins the underline *width*; here the
/// primary span covers the `中` grapheme (3 bytes, 2 cells) and the secondary
/// label sits on `b` *after* it, so its connector must be offset by the extra
/// display cell — a byte- or char-counted column would land one space short.
#[test]
fn html_cjk_columns() {
    const SRC: &str = "set a = \"中\" + b;\n";
    let diag = Diagnostic::error("wide char")
        .with_code_str("E030")
        .with_span(Span::new(FileId::DUMMY, 9, 12)) // the `中` grapheme
        .with_secondary_label(Span::new(FileId::DUMMY, 16, 17), "after CJK") // `b`
        .with_source(SRC);
    let opts = DiagOpts::default();
    let mut r = HtmlRenderer::new();
    let out = r.render(&diag, &ctx(&opts));
    insta::assert_snapshot!("html_cjk_columns", out);
}

/// JSON object.  Round-trip through `serde_json` first so a broken renderer
/// fails with a parse error rather than a useless text diff.
#[test]
fn json_kitchen_sink() {
    let opts = DiagOpts::default();
    let mut r = JsonRenderer::new_pretty();
    let out = r.render(&kitchen_sink(), &ctx(&opts));
    let _: serde_json::Value = serde_json::from_str(&out).expect("valid JSON");
    insta::assert_snapshot!("json_kitchen_sink", out);
}

/// Summary-line pluralization, all combinations in one snapshot so a
/// translated-string change is visible.
#[test]
fn summary_shapes() {
    // English-sensitive: `Lang` is a process-global `AtomicU8`, so hold the
    // shared test lock against concurrent `Lang::set` users.
    let _lang_guard = crate::diagnostics::i18n::acquire_test_lang_lock();
    let r = GlyphRenderer::new(false);
    let out = [
        r.render_summary(0, 0),
        r.render_summary(1, 0),
        r.render_summary(2, 0),
        r.render_summary(0, 1),
        r.render_summary(0, 4),
        r.render_summary(2, 1),
    ]
    .join("\n");
    insta::assert_snapshot!("summary_shapes", out);
}

/// Same shape assertions as [`summary_shapes`] but with colour enabled, so
/// the severity-driven accent colour is pinned: error-containing summaries
/// must be painted with the error style, warning-only summaries with the
/// warning style.
#[test]
fn summary_shapes_colored() {
    let _lang_guard = crate::diagnostics::i18n::acquire_test_lang_lock();
    let r = GlyphRenderer::new(true);
    let out = [
        r.render_summary(0, 0),
        r.render_summary(1, 0),
        r.render_summary(2, 0),
        r.render_summary(0, 1),
        r.render_summary(0, 4),
        r.render_summary(2, 1),
    ]
    .join("\n");
    insta::assert_snapshot!("summary_shapes_colored", out);
}
