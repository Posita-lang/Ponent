//! Keyword-as-identifier policy for path position.
//!
//! Posita keywords are NOT reserved in path position. Any keyword token
//! may appear after `::` — e.g. `T::default()`, `T::move()`.
//!
//! When adding a new keyword, see the doc comment on
//! [`Token::as_ident_symbol`].

use crate::symbol::Symbol;

use super::token::Token;

impl Token {
    /// If this token represents an identifier — either a true `Ident` or a
    /// keyword that can serve as a name in path position (e.g. `default`
    /// after `::`) — return the corresponding `Symbol`.
    ///
    /// # Track‑A Policy (Engineering Canon)
    /// This is an intentionally comprehensive mapping: **every keyword that
    /// could plausibly appear as a type / method / variant name after `::`
    /// is accepted**.  This matches rustc's approach where keywords are
    /// weak identifiers in path position.
    ///
    /// ## When adding a new keyword token
    /// 1. Add the token variant to the `#[token("keyword")]` section of
    ///    `Token` (the logos lexer).
    /// 2. Add an arm to **this method** so the keyword is accepted after
    ///    `::` in paths, patterns, and expressions.
    /// 3. Add a test — at minimum a parser test that exercises the keyword
    ///    in a qualified path (e.g. `Mod::keyword`).
    /// 4. **Do NOT** add a keyword here if it can never be a legal
    ///    identifier (e.g. `;`, `{`, `::`, `->` are structural tokens
    ///    with their own `Token` variants and are not keywords).
    /// 5. **Do NOT** skip a keyword just because it "seems unlikely" to be
    ///    a method name — `default`, `move`, `copy`, and `type` have all
    ///    been needed.  Missing one will produce a confusing
    ///    `expected identifier after '::'` error.
    pub fn as_ident_symbol(&self) -> Option<Symbol> {
        match self {
            Token::Ident(s) => Some(*s),
            // Keywords commonly used as identifiers in paths / methods.
            Token::Default => Some(Symbol::intern("default")),
            Token::Move => Some(Symbol::intern("move")),
            Token::Copy => Some(Symbol::intern("copy")),
            Token::Ref => Some(Symbol::intern("ref")),
            Token::Mut => Some(Symbol::intern("mut")),
            Token::Type => Some(Symbol::intern("type")),
            Token::SelfKw => Some(Symbol::intern("Self")),
            Token::Async => Some(Symbol::intern("async")),
            Token::Await => Some(Symbol::intern("await")),
            Token::Catch => Some(Symbol::intern("catch")),
            Token::Let => Some(Symbol::intern("let")),
            Token::Where => Some(Symbol::intern("where")),
            Token::When => Some(Symbol::intern("when")),
            Token::As => Some(Symbol::intern("as")),
            Token::In => Some(Symbol::intern("in")),
            Token::And => Some(Symbol::intern("and")),
            Token::Or => Some(Symbol::intern("or")),
            Token::Not => Some(Symbol::intern("not")),
            Token::Isolate => Some(Symbol::intern("isolate")),
            Token::Pure => Some(Symbol::intern("pure")),
            Token::Io => Some(Symbol::intern("io")),
            Token::Trusted => Some(Symbol::intern("trusted")),
            Token::Const => Some(Symbol::intern("const")),
            Token::Ghost => Some(Symbol::intern("ghost")),
            Token::Layout => Some(Symbol::intern("layout")),
            Token::Validate => Some(Symbol::intern("validate")),
            Token::Exists => Some(Symbol::intern("exists")),
            Token::Forall => Some(Symbol::intern("forall")),
            Token::Implies => Some(Symbol::intern("implies")),
            Token::On => Some(Symbol::intern("on")),
            Token::Trait => Some(Symbol::intern("trait")),
            Token::Impl => Some(Symbol::intern("impl")),
            Token::Cfg => Some(Symbol::intern("cfg")),
            Token::Hint => Some(Symbol::intern("hint")),
            Token::Old => Some(Symbol::intern("old")),
            Token::Diverges => Some(Symbol::intern("diverges")),
            Token::Overrides => Some(Symbol::intern("overrides")),
            Token::Propagates => Some(Symbol::intern("propagates")),
            Token::Poly => Some(Symbol::intern("poly")),
            Token::Unbox => Some(Symbol::intern("unbox")),
            Token::Extern => Some(Symbol::intern("extern")),
            Token::Pub => Some(Symbol::intern("pub")),
            Token::Unsafe => Some(Symbol::intern("unsafe")),
            Token::Panic => Some(Symbol::intern("panic")),
            Token::Finally => Some(Symbol::intern("finally")),
            Token::Dyn => Some(Symbol::intern("dyn")),
            Token::By => Some(Symbol::intern("by")),
            Token::Wrap => Some(Symbol::intern("wrap")),
            Token::Saturate => Some(Symbol::intern("saturate")),
            Token::Trap => Some(Symbol::intern("trap")),
            Token::Round => Some(Symbol::intern("round")),
            Token::Trunc => Some(Symbol::intern("trunc")),
            Token::Ceil => Some(Symbol::intern("ceil")),
            Token::Floor => Some(Symbol::intern("floor")),
            Token::Ieee => Some(Symbol::intern("ieee")),
            // The remaining SYNTAX.md keywords were missing
            // from the path-position table — any keyword is valid
            // after `::` (`T::default()`, `T::move()`),
            // so every keyword with a dedicated token must resolve to its
            // surface name in path position (e.g. `T::requires`).
            Token::Def => Some(Symbol::intern("def")),
            Token::Set => Some(Symbol::intern("set")),
            Token::With => Some(Symbol::intern("with")),
            Token::Return => Some(Symbol::intern("return")),
            Token::If => Some(Symbol::intern("if")),
            Token::Else => Some(Symbol::intern("else")),
            Token::For => Some(Symbol::intern("for")),
            Token::While => Some(Symbol::intern("while")),
            Token::Loop => Some(Symbol::intern("loop")),
            Token::Leave => Some(Symbol::intern("leave")),
            Token::Continue => Some(Symbol::intern("continue")),
            Token::Comptime => Some(Symbol::intern("comptime")),
            Token::Generate => Some(Symbol::intern("generate")),
            Token::Import => Some(Symbol::intern("import")),
            Token::From => Some(Symbol::intern("from")),
            Token::True => Some(Symbol::intern("true")),
            Token::False => Some(Symbol::intern("false")),
            Token::Auto => Some(Symbol::intern("auto")),
            Token::Sizeof => Some(Symbol::intern("sizeof")),
            Token::Alignof => Some(Symbol::intern("alignof")),
            Token::Requires => Some(Symbol::intern("requires")),
            Token::Ensures => Some(Symbol::intern("ensures")),
            Token::Invariant => Some(Symbol::intern("invariant")),
            Token::Constraint => Some(Symbol::intern("constraint")),
            Token::NoDefault => Some(Symbol::intern("no_default")),
            Token::Edition => Some(Symbol::intern("edition")),
            Token::Deprecated => Some(Symbol::intern("deprecated")),
            Token::Experimental => Some(Symbol::intern("experimental")),
            Token::Endian => Some(Symbol::intern("endian")),
            Token::BitOrder => Some(Symbol::intern("bit_order")),
            Token::Align => Some(Symbol::intern("align")),
            Token::Pad => Some(Symbol::intern("pad")),
            Token::Packed => Some(Symbol::intern("packed")),
            Token::Task => Some(Symbol::intern("task")),
            Token::Channel => Some(Symbol::intern("channel")),
            Token::Linear => Some(Symbol::intern("linear")),
            Token::Consume => Some(Symbol::intern("consume")),
            Token::ScopeCleanup => Some(Symbol::intern("scope_cleanup")),
            Token::Trigger => Some(Symbol::intern("trigger")),
            Token::MissingMatch => Some(Symbol::intern("missing_match")),
            Token::ApplyLemma => Some(Symbol::intern("apply_lemma")),
            Token::Decreases => Some(Symbol::intern("decreases")),
            Token::Terminates => Some(Symbol::intern("terminates")),
            Token::MustUse => Some(Symbol::intern("must_use")),
            Token::MustHandle => Some(Symbol::intern("must_handle")),
            Token::LinkProof => Some(Symbol::intern("link_proof")),
            Token::Exhaustive => Some(Symbol::intern("exhaustive")),
            Token::NoAllocError => Some(Symbol::intern("no_alloc_error")),
            Token::NoPanic => Some(Symbol::intern("no_panic")),
            Token::DebugInfo => Some(Symbol::intern("debug_info")),
            Token::IeeeContracts => Some(Symbol::intern("ieee_contracts")),
            Token::AuditLog => Some(Symbol::intern("audit_log")),
            Token::Interrupt => Some(Symbol::intern("interrupt")),
            Token::Match => Some(Symbol::intern("match")),
            Token::OnTimeout => Some(Symbol::intern("on_timeout")),
            Token::OnCancel => Some(Symbol::intern("on_cancel")),
            _ => None,
        }
    }
}
