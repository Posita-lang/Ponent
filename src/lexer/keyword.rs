//! Keyword-as-identifier policy for path position.
//!
//! Posita keywords are NOT reserved in path position. Any keyword token
//! may appear after `::` — e.g. `T::default()`, `T::move()`.
//!
//! Each keyword's `Symbol` is interned exactly once per process into a
//! `LazyLock` static (see [`keyword_symbols!`]); the match arms below
//! return a copy of that static, so a path-position lookup costs one
//! atomic load instead of a `RwLock`-guarded hash lookup on every call.
//!
//! When adding a new keyword, see the doc comment on
//! [`Token::as_ident_symbol`] (generated inside the macro below).

use std::sync::LazyLock;

use crate::symbol::Symbol;

use super::token::Token;

/// Single source of truth for keyword→Symbol resolution.
///
/// One invocation lists every keyword that may serve as a name in path
/// position as `SYM_NAME : Variant => "surface";` and generates BOTH:
///   1. a `LazyLock<Symbol>` static per keyword (interned once, lazily),
///   2. the corresponding match arm in `Token::as_ident_symbol`.
///
/// Keep the list alphabetical. `Ro` and `Codomain` are deliberately
/// absent (reserved words — see rule 4 in the method doc); structural
/// tokens and literal payload variants fall into the `_ => None` arm.
macro_rules! keyword_symbols {
    ($($name:ident : $variant:ident => $surface:literal;)*) => {
        $(
            static $name: LazyLock<Symbol> = LazyLock::new(|| Symbol::intern($surface));
        )*

        impl<'src> Token<'src> {
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
            /// 2. Add ONE entry to the `keyword_symbols!` invocation below — it
            ///    generates the `LazyLock` static and the match arm together.
            ///    Also add an arm to `Token::to_user_string` in `display.rs` —
            ///    that match *is* exhaustive, so the compiler catches a missing
            ///    display arm at compile time; this match is not, and is guarded
            ///    instead by the `every_keyword_resolves_in_path_position` test
            ///    in `tests.rs`.
            /// 3. Add a test — at minimum a parser test that exercises the keyword
            ///    in a qualified path (e.g. `Mod::keyword`).
            /// 4. **Do NOT** add a keyword here if it can never be a legal
            ///    identifier (e.g. `;`, `{`, `::`, `->` are structural tokens
            ///    with their own `Token` variants and are not keywords).  The
            ///    reserved words `ro` and `codomain` (SYNTAX.md) are likewise
            ///    absent — pinned by `reserved_words_are_never_identifiers`.
            /// 5. Conversely, words NOT in SYNTAX.md's keyword list — `from`,
            ///    `on`, `on_timeout`, `on_cancel`, `round`, `trunc`, `ceil`,
            ///    `floor` — have NO dedicated token: they lex as `Ident` and the
            ///    parser matches them contextually inside the productions that
            ///    use them (`ensures on Ok/Err`,
            ///    `as Int<32> round`).  They resolve via the `Token::Ident` arm;
            ///    do not "restore" them as reserved tokens (pinned by
            ///    `contextual_keywords_are_plain_identifiers`).
            /// 6. **Do NOT** skip a keyword just because it "seems unlikely" to be
            ///    a method name — `default`, `move`, `copy`, and `type` have all
            ///    been needed.  Missing one will produce a confusing
            ///    `expected identifier after '::'` error.
            #[inline]
            pub fn as_ident_symbol(&self) -> Option<Symbol> {
                match self {
                    Token::Ident(s) => Some(*s),
                    $(
                        Token::$variant => Some(*$name),
                    )*
                    _ => None,
                }
            }
        }
    };
}

keyword_symbols! {
    SYM_SELF_KW : SelfKw => "Self";
    SYM_ALIGN : Align => "align";
    SYM_ALIGNOF : Alignof => "alignof";
    SYM_AND : And => "and";
    SYM_APPLY_LEMMA : ApplyLemma => "apply_lemma";
    SYM_AS : As => "as";
    SYM_ASYNC : Async => "async";
    SYM_AUDIT_LOG : AuditLog => "audit_log";
    SYM_AUTO : Auto => "auto";
    SYM_AWAIT : Await => "await";
    SYM_BIT_ORDER : BitOrder => "bit_order";
    SYM_BY : By => "by";
    SYM_CATCH : Catch => "catch";
    SYM_CFG : Cfg => "cfg";
    SYM_CHANNEL : Channel => "channel";
    SYM_COMPTIME : Comptime => "comptime";
    SYM_CONST : Const => "const";
    SYM_CONSTRAINT : Constraint => "constraint";
    SYM_CONSUME : Consume => "consume";
    SYM_CONTINUE : Continue => "continue";
    SYM_COPY : Copy => "copy";
    SYM_DEBUG_INFO : DebugInfo => "debug_info";
    SYM_DECREASES : Decreases => "decreases";
    SYM_DEF : Def => "def";
    SYM_DEFAULT : Default => "default";
    SYM_DEPRECATED : Deprecated => "deprecated";
    SYM_DIVERGES : Diverges => "diverges";
    SYM_DYN : Dyn => "dyn";
    SYM_EDITION : Edition => "edition";
    SYM_ELSE : Else => "else";
    SYM_ENDIAN : Endian => "endian";
    SYM_ENSURES : Ensures => "ensures";
    SYM_EXHAUSTIVE : Exhaustive => "exhaustive";
    SYM_EXISTS : Exists => "exists";
    SYM_EXPERIMENTAL : Experimental => "experimental";
    SYM_EXTERN : Extern => "extern";
    SYM_FALSE : False => "false";
    SYM_FINALLY : Finally => "finally";
    SYM_FOR : For => "for";
    SYM_FORALL : Forall => "forall";
    SYM_GENERATE : Generate => "generate";
    SYM_GHOST : Ghost => "ghost";
    SYM_HINT : Hint => "hint";
    SYM_IEEE : Ieee => "ieee";
    SYM_IEEE_CONTRACTS : IeeeContracts => "ieee_contracts";
    SYM_IF : If => "if";
    SYM_IMPL : Impl => "impl";
    SYM_IMPLIES : Implies => "implies";
    SYM_IMPORT : Import => "import";
    SYM_IN : In => "in";
    SYM_INTERRUPT : Interrupt => "interrupt";
    SYM_INVARIANT : Invariant => "invariant";
    SYM_IO : Io => "io";
    SYM_ISOLATE : Isolate => "isolate";
    SYM_LAYOUT : Layout => "layout";
    SYM_LEAVE : Leave => "leave";
    SYM_LET : Let => "let";
    SYM_LINEAR : Linear => "linear";
    SYM_LINK_PROOF : LinkProof => "link_proof";
    SYM_LOOP : Loop => "loop";
    SYM_MATCH : Match => "match";
    SYM_MISSING_MATCH : MissingMatch => "missing_match";
    SYM_MOVE : Move => "move";
    SYM_MUST_HANDLE : MustHandle => "must_handle";
    SYM_MUST_USE : MustUse => "must_use";
    SYM_MUT : Mut => "mut";
    SYM_NO_ALLOC_ERROR : NoAllocError => "no_alloc_error";
    SYM_NO_DEFAULT : NoDefault => "no_default";
    SYM_NO_PANIC : NoPanic => "no_panic";
    SYM_NOT : Not => "not";
    SYM_OLD : Old => "old";
    SYM_OR : Or => "or";
    SYM_OVERRIDES : Overrides => "overrides";
    SYM_PACKED : Packed => "packed";
    SYM_PAD : Pad => "pad";
    SYM_PANIC : Panic => "panic";
    SYM_POLY : Poly => "poly";
    SYM_PROPAGATES : Propagates => "propagates";
    SYM_PUB : Pub => "pub";
    SYM_PURE : Pure => "pure";
    SYM_REF : Ref => "ref";
    SYM_REQUIRES : Requires => "requires";
    SYM_RETURN : Return => "return";
    SYM_SATURATE : Saturate => "saturate";
    SYM_SCOPE_CLEANUP : ScopeCleanup => "scope_cleanup";
    SYM_SET : Set => "set";
    SYM_SIZEOF : Sizeof => "sizeof";
    SYM_TASK : Task => "task";
    SYM_TERMINATES : Terminates => "terminates";
    SYM_TRAIT : Trait => "trait";
    SYM_TRAP : Trap => "trap";
    SYM_TRIGGER : Trigger => "trigger";
    SYM_TRUE : True => "true";
    SYM_TRUSTED : Trusted => "trusted";
    SYM_TYPE : Type => "type";
    SYM_UNBOX : Unbox => "unbox";
    SYM_UNSAFE : Unsafe => "unsafe";
    SYM_VALIDATE : Validate => "validate";
    SYM_WHEN : When => "when";
    SYM_WHERE : Where => "where";
    SYM_WHILE : While => "while";
    SYM_WITH : With => "with";
    SYM_WRAP : Wrap => "wrap";
}
