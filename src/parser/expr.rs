//! Pratt‑style expression parser with prefix/infix handling and all expression constructs.

use super::*;
use crate::ast::*;
use crate::diagnostics::Diagnostic;
use crate::diagnostics::ErrCode;
use crate::lexer::Token;
use crate::lexer::literal::{
    parse_byte_char_literal, parse_byte_string_literal, parse_char_literal, parse_float_literal,
    parse_int_literal, parse_string_literal,
};
use crate::symbol::Symbol;
use smallvec::smallvec;

impl<'src, 'input> Parser<'src, 'input> {
    // -------------------------------------------------------------------------
    // Expression entry
    // -------------------------------------------------------------------------

    pub fn parse_expr(
        &mut self,
        bump: &'input bumpalo::Bump,
    ) -> Result<Expr<'input>, Diagnostic<'src>> {
        if let Some(e) = self.parse_expr_fast(bump)? {
            return Ok(e);
        }
        self.parse_expr_bp(0, bump)
    }

    /// Fast path for plain literals followed by a terminator.
    /// Hot path (the industrial-engineering pass): the overwhelming
    /// majority of expressions are plain literals — build them directly
    /// without the full Pratt machinery.  A following `:` (type
    /// annotation) or anything else falls back to the general parser.
    #[inline(always)]
    fn parse_expr_fast(
        &mut self,
        bump: &'input bumpalo::Bump,
    ) -> Result<Option<Expr<'input>>, Diagnostic<'src>> {
        // Check the token AFTER the literal FIRST (without consuming):
        // only a terminator (or a type-annotation colon) lets the literal
        // stand alone — otherwise (e.g. `1 + 2`) we must fall back WITHOUT
        // having consumed anything, or the general parser would start
        // mid-expression.
        // NOTE: Colon is deliberately NOT a terminator here — `1: T` is a
        // literal with a type annotation, which the general parser must
        // handle (fall back without consuming).
        if !matches!(
            self.peek_next(),
            Some(Token::Semicolon)
                | Some(Token::RParen)
                | Some(Token::Comma)
                | Some(Token::RBracket)
                | Some(Token::RBrace)
                | None
        ) {
            return Ok(None);
        }
        match self.peek() {
            Some(Token::IntLiteral(_))
            | Some(Token::HexLiteral(_))
            | Some(Token::BinLiteral(_))
            | Some(Token::FloatLiteral(_))
            | Some(Token::StringLiteral(_))
            | Some(Token::ByteStringLiteral(_))
            | Some(Token::CharLiteral(_))
            | Some(Token::ByteCharLiteral(_))
            | Some(Token::True)
            | Some(Token::False) => {
                let expr = self.parse_literal(bump)?;
                Ok(Some(expr))
            }
            _ => Ok(None),
        }
    }

    // -------------------------------------------------------------------------
    // Pratt parser (binding power)
    // -------------------------------------------------------------------------

    fn parse_expr_bp(
        &mut self,
        min_bp: u8,
        bump: &'input bumpalo::Bump,
    ) -> Result<Expr<'input>, Diagnostic<'src>> {
        self.recursion_depth += 1;
        if self.recursion_depth > self.max_recursion_depth {
            self.recursion_depth -= 1;
            return Err(Diagnostic::error(format!("maximum recursion depth {} exceeded", self.max_recursion_depth))
                .with_code_str(ErrCode::E006)
                .with_help("the parser reached its recursion limit — the type/structure may be deeply nested or self-referential")
                .with_suggestion("try breaking up deeply nested structures, or use less complex type expressions")
                .with_span(self.span()));
        }
        let result = self.parse_expr_bp_inner(min_bp, bump);
        self.recursion_depth -= 1;
        result
    }

    fn parse_expr_bp_inner(
        &mut self,
        min_bp: u8,
        bump: &'input bumpalo::Bump,
    ) -> Result<Expr<'input>, Diagnostic<'src>> {
        let mut lhs = self.parse_prefix(bump)?;
        loop {
            // Borrow the peeked token directly — no clone needed;
            // prefix_binding_power is a static fn that doesn't read &self.
            let lbp = match self.peek() {
                Some(tok)
                    if !matches!(
                        tok,
                        Token::Semicolon
                            | Token::RBrace
                            | Token::RParen
                            | Token::Comma
                            | Token::Colon
                            | Token::In
                    ) =>
                {
                    match Self::prefix_binding_power(Some(tok)) {
                        Some((lbp, _)) => lbp,
                        None => break,
                    }
                }
                _ => break,
            };
            if lbp < min_bp {
                break;
            }
            if self.restrictions.contains(ParseRestrictions::NO_COMPARISON) {
                let is_compare = match self.peek() {
                    Some(tok) => crate::lexer::token_class(tok)
                        .contains(crate::lexer::TokenClass::COMPARISON),
                    None => false,
                };
                if is_compare {
                    break;
                }
            }
            lhs = self.parse_infix(lhs, lbp, bump)?;
        }
        Ok(lhs)
    }

    fn prefix_binding_power(token: Option<&Token>) -> Option<(u8, bool)> {
        match token {
            // Mul/Div/Rem: precedence 1 (highest per SYNTAX.md)
            Some(Token::Star) | Some(Token::Slash) | Some(Token::Percent) => Some((15, true)),
            // Add/Sub: precedence 2
            Some(Token::Plus) | Some(Token::Minus) => Some((13, true)),
            // Wrap/saturate/trap variants follow their base operator's precedence
            Some(Token::StarWrap) | Some(Token::StarSaturate) | Some(Token::StarTrap) => {
                Some((15, true))
            }
            Some(Token::PlusWrap)
            | Some(Token::MinusWrap)
            | Some(Token::PlusSaturate)
            | Some(Token::MinusSaturate)
            | Some(Token::PlusTrap)
            | Some(Token::MinusTrap) => Some((13, true)),
            // Bitwise shift: precedence 3
            Some(Token::Shl) | Some(Token::Shr) => Some((12, true)),
            // Bitwise and: precedence 4
            Some(Token::Ampersand) => Some((11, true)),
            // Bitwise xor: precedence 5
            Some(Token::Caret) => Some((10, true)),
            // Bitwise or: precedence 6
            Some(Token::Pipe) => Some((9, true)),
            // Comparison: precedence 7
            Some(Token::EqEq) | Some(Token::Neq) | Some(Token::Lt) | Some(Token::Gt)
            | Some(Token::Le) | Some(Token::Ge) => Some((8, true)),
            // Logical and: precedence 8
            Some(Token::And) => Some((7, true)),
            // Logical or: precedence 9
            Some(Token::Or) => Some((6, true)),
            // Range: precedence 11 (lowest)
            Some(Token::DotDot) | Some(Token::DotDotEq) => Some((1, true)),
            // Postfix / access / call operators (bind tightest)
            Some(Token::LParen)
            | Some(Token::LBracket)
            | Some(Token::Dot)
            | Some(Token::Apostrophe) => Some((18, true)),
            Some(Token::Question) => Some((17, true)),
            // Postfix comptime call marker: `func!()`.
            Some(Token::Bang) => Some((16, true)),
            // `not` and `~` are prefix-only; they are handled in parse_prefix
            // and must NOT appear in prefix_binding_power — otherwise the infix
            // loop would treat them as valid infix operators and call parse_infix,
            // which has no arm for them and hits unreachable!().
            // Some(Token::Not) => Some((16, false)),
            // Some(Token::Tilde) => Some((16, false)),
            // Cast
            Some(Token::As) => Some((14, true)),
            // Catch expression
            Some(Token::Catch) => Some((1, true)),
            _ => None,
        }
    }

    // -------------------------------------------------------------------------
    // Prefix expression parser
    // -------------------------------------------------------------------------

    fn parse_prefix(
        &mut self,
        bump: &'input bumpalo::Bump,
    ) -> Result<Expr<'input>, Diagnostic<'src>> {
        let start = self.span().start;
        match self.peek() {
            Some(Token::IntLiteral(_))
            | Some(Token::HexLiteral(_))
            | Some(Token::BinLiteral(_))
            | Some(Token::FloatLiteral(_))
            | Some(Token::StringLiteral(_))
            | Some(Token::ByteStringLiteral(_))
            | Some(Token::CharLiteral(_))
            | Some(Token::ByteCharLiteral(_))
            | Some(Token::True)
            | Some(Token::False) => {
                let expr = self.parse_literal(bump)?;
                if matches!(self.peek(), Some(Token::Colon)) {
                    self.advance();
                    let ty = self.parse_type(bump)?;
                    let end = self.span().end;
                    Ok(Expr::TypeAnnotated {
                        expr: Self::alloc_shared(bump, expr),
                        ty: Self::alloc_shared(bump, ty),
                        span: Span::new(self.file_id, start, end),
                    })
                } else {
                    Ok(expr)
                }
            }
            Some(Token::Ident(_)) => self.parse_path_or_literal(start, bump),
            Some(Token::LParen) => {
                self.advance();
                if matches!(self.peek(), Some(Token::RParen)) {
                    self.advance();
                    let end = self.span().end;
                    return Ok(Expr::Tuple(
                        Self::alloc_slice(bump, &[]),
                        Span::new(self.file_id, start, end),
                    ));
                }
                let expr = self.parse_expr(bump)?;
                if matches!(self.peek(), Some(Token::Colon)) {
                    self.advance();
                    let ty = self.parse_type(bump)?;
                    self.expect(Token::RParen)?;
                    let end = self.span().end;
                    Ok(Expr::TypeAnnotated {
                        expr: Self::alloc_shared(bump, expr),
                        ty: Self::alloc_shared(bump, ty),
                        span: Span::new(self.file_id, start, end),
                    })
                } else if matches!(self.peek(), Some(Token::Comma)) {
                    let mut exprs = bumpalo::collections::Vec::with_capacity_in(4, bump);
                    exprs.push(expr);
                    while matches!(self.peek(), Some(Token::Comma)) {
                        self.advance();
                        exprs.push(self.parse_expr(bump)?);
                    }
                    self.expect(Token::RParen)?;
                    let end = self.span().end;
                    Ok(Expr::Tuple(
                        exprs.into_bump_slice(),
                        Span::new(self.file_id, start, end),
                    ))
                } else {
                    self.expect(Token::RParen)?;
                    Ok(expr)
                }
            }
            Some(Token::LBracket) => {
                self.advance();
                let mut exprs = bumpalo::collections::Vec::with_capacity_in(8, bump);
                loop {
                    if matches!(self.peek(), Some(Token::RBracket)) {
                        self.advance();
                        break;
                    }
                    exprs.push(self.parse_expr(bump)?);
                    if matches!(self.peek(), Some(Token::Comma)) {
                        self.advance();
                    } else {
                        self.expect(Token::RBracket)?;
                        break;
                    }
                }
                let end = self.span().end;
                Ok(Expr::Array(
                    exprs.into_bump_slice(),
                    Span::new(self.file_id, start, end),
                ))
            }
            Some(Token::Plus) | Some(Token::Minus) | Some(Token::Star) | Some(Token::Slash)
            | Some(Token::Percent) => {
                let next = self.peek_next();
                let is_operator_arg = matches!(
                    next,
                    Some(Token::Comma)
                        | Some(Token::RParen)
                        | Some(Token::RBracket)
                        | Some(Token::RBrace)
                );
                if is_operator_arg {
                    let op_tok = self.advance().ok_or_else(|| {
                        Diagnostic::error("unexpected token")
                            .with_code_str(ErrCode::E003)
                            .with_help("expected an expression after the operator position")
                            .with_span(Span::new(self.file_id, 0, 0))
                    })?;
                    let op_name = match op_tok {
                        Token::Plus => Symbol::intern("+"),
                        Token::Minus => Symbol::intern("-"),
                        Token::Star => Symbol::intern("*"),
                        Token::Slash => Symbol::intern("/"),
                        Token::Percent => Symbol::intern("%"),
                        _ => unreachable!(),
                    };
                    let end = self.span().end;
                    Ok(Expr::Ident(op_name, Span::new(self.file_id, start, end)))
                } else {
                    match self.advance().ok_or_else(|| Diagnostic::error("unexpected token")
                        .with_code_str(ErrCode::E003)
                        .with_help("expected a unary operator (`-`, `*`, `!`, `~`, `&`, `move`)")
                        .with_span(Span::new(self.file_id, 0, 0)))?
                    {
                        Token::Minus => {
                            let expr = self.parse_prefix(bump)?;
                            let end = self.span().end;
                            Ok(Expr::UnaryOp { op: UnaryOp::Neg, expr: Self::alloc_shared(bump, expr), span: Span::new(self.file_id, start, end) })
                        }
                        Token::Star => {
                            let expr = self.parse_prefix(bump)?;
                            let end = self.span().end;
                            Ok(Expr::UnaryOp { op: UnaryOp::Deref, expr: Self::alloc_shared(bump, expr), span: Span::new(self.file_id, start, end) })
                        }
                        _ => Err(Diagnostic::error("unexpected operator in expression")
                            .with_code_str(ErrCode::E007)
                            .with_help("this operator is not valid at this position — check for missing operands or extra operators")
                            .with_span(self.span())),
                    }
                }
            }
            Some(Token::If) => self.parse_if_expr(bump),
            Some(Token::Task) => {
                let start = self.span().start;
                self.advance();
                self.expect(Token::LBrace)?;
                let body = self.parse_block(bump)?;
                self.expect(Token::RBrace)?;
                let end = self.span().end;
                Ok(Expr::Task {
                    body,
                    span: Span::new(self.file_id, start, end),
                })
            }
            Some(Token::Match) => self.parse_match_expr(bump),
            Some(Token::Leave) => {
                self.advance();
                self.expect(Token::With)?;
                let expr = self.parse_expr(bump)?;
                let end = self.span().end;
                Ok(Expr::LeaveWith {
                    expr: Self::alloc_shared(bump, expr),
                    is_return: false,
                    span: Span::new(self.file_id, start, end),
                })
            }
            Some(Token::Return) => {
                // Consume the `return` keyword FIRST — the value
                // expression below is parsed from the NEXT token;
                // otherwise `parse_expr` on the value re-enters this
                // branch with the unconsumed `return` (an infinite
                // recursion — stack overflow — exposed by expression-
                // position returns such as a match arm's `return 0`).
                self.advance();
                let value = if !matches!(self.peek(), Some(Token::Semicolon) | Some(Token::RBrace))
                {
                    Some(Self::alloc_shared(bump, self.parse_expr(bump)?))
                } else {
                    None
                };
                let end = self.span().end;
                Ok(Expr::LeaveWith {
                    expr: value.unwrap_or_else(|| {
                        Self::alloc_shared(
                            bump,
                            Expr::Tuple(
                                Self::alloc_slice(bump, &[]),
                                Span::new(self.file_id, start, end),
                            ),
                        )
                    }),
                    is_return: true,
                    span: Span::new(self.file_id, start, end),
                })
            }
            Some(Token::Await) => {
                self.advance();
                let expr = self.parse_expr(bump)?;
                let end = self.span().end;
                Ok(Expr::Await {
                    expr: Self::alloc_shared(bump, expr),
                    span: Span::new(self.file_id, start, end),
                })
            }
            Some(Token::Poly) => {
                self.advance();
                self.expect(Token::LParen)?;
                let expr = self.parse_expr(bump)?;
                self.expect(Token::RParen)?;
                let end = self.span().end;
                Ok(Expr::PolyBox {
                    expr: Self::alloc_shared(bump, expr),
                    scheme: None,
                    span: Span::new(self.file_id, start, end),
                })
            }
            Some(Token::Unbox) => {
                self.advance();
                self.expect(Token::LParen)?;
                let expr = self.parse_expr(bump)?;
                self.expect(Token::RParen)?;
                let end = self.span().end;
                Ok(Expr::PolyUnbox {
                    expr: Self::alloc_shared(bump, expr),
                    scheme: None,
                    span: Span::new(self.file_id, start, end),
                })
            }
            Some(Token::Old) => {
                self.advance();
                self.expect(Token::LParen)?;
                let expr = self.parse_expr(bump)?;
                self.expect(Token::RParen)?;
                let end = self.span().end;
                Ok(Expr::Old(
                    Self::alloc_shared(bump, expr),
                    Span::new(self.file_id, start, end),
                ))
            }
            // `codomain` is the reserved return-value name in `ensures`
            // clauses (SYNTAX.md).  It parses to a plain identifier
            // expression; the checker binds that symbol to the return type.
            Some(Token::Codomain) => {
                self.advance();
                let end = self.span().end;
                Ok(Expr::Ident(
                    crate::symbol::Symbol::intern("codomain"),
                    Span::new(self.file_id, start, end),
                ))
            }
            Some(Token::Not) => {
                self.advance();
                let expr = self.parse_expr_bp(1, bump)?;
                let end = self.span().end;
                Ok(Expr::UnaryOp {
                    op: UnaryOp::Not,
                    expr: Self::alloc_shared(bump, expr),
                    span: Span::new(self.file_id, start, end),
                })
            }
            Some(Token::Tilde) => {
                self.advance();
                let expr = self.parse_prefix(bump)?;
                let end = self.span().end;
                Ok(Expr::UnaryOp {
                    op: UnaryOp::BitNot,
                    expr: Self::alloc_shared(bump, expr),
                    span: Span::new(self.file_id, start, end),
                })
            }
            Some(Token::Ampersand) => {
                self.advance();
                // `&ro` freezes a `&mut T` into a `&T` (SYNTAX.md §Reference
                // Coercion): `ro` only has special meaning immediately
                // following `&` in a borrow expression.  `ro` is a reserved
                // token, so anywhere else the parser rejects it with a
                // proper "unexpected `ro`" diagnostic instead of silently
                // treating it as an identifier.
                let read_only = matches!(self.peek(), Some(Token::Ro));
                if read_only {
                    self.advance();
                }
                let mutable = !read_only && matches!(self.peek(), Some(Token::Mut));
                if mutable {
                    self.advance();
                }
                // The borrow's operand must include the POSTFIX chain
                // (`&mut arr[0]` = `&mut (arr[0])`, NOT `(&mut arr)[0]`):
                // parse at binding power 18 (= the postfix operators'
                // binding power, above every binary operator) so the
                // borrow wraps the whole index/field/call — mirroring
                // rustc's `parse_expr_borrow` → `parse_expr_prefix`.
                let expr = self.parse_expr_bp(18, bump)?;
                let end = self.span().end;
                Ok(Expr::UnaryOp {
                    op: if read_only {
                        UnaryOp::Ro
                    } else if mutable {
                        UnaryOp::RefMut
                    } else {
                        UnaryOp::Ref
                    },
                    expr: Self::alloc_shared(bump, expr),
                    span: Span::new(self.file_id, start, end),
                })
            }
            Some(Token::Move) => {
                self.advance();
                let expr = self.parse_prefix(bump)?;
                let end = self.span().end;
                Ok(Expr::Move(
                    Self::alloc_shared(bump, expr),
                    Span::new(self.file_id, start, end),
                ))
            }
            Some(Token::Unsafe) => {
                self.advance();
                self.expect(Token::LBrace)?;
                let body = self.parse_block(bump)?;
                self.expect(Token::RBrace)?;
                let end = self.span().end;
                Ok(Expr::UnsafeBlock {
                    body,
                    span: Span::new(self.file_id, start, end),
                })
            }
            Some(Token::Pipe) => self.parse_closure(start, bump),
            Some(Token::LBrace) => {
                if self
                    .restrictions
                    .contains(ParseRestrictions::NO_STRUCT_LITERAL)
                {
                    self.advance();
                    let body = self.parse_block(bump)?;
                    self.expect(Token::RBrace)?;
                    Ok(Expr::Block(
                        body,
                        Span::new(self.file_id, start, self.span().end),
                    ))
                } else {
                    self.parse_struct_lit(vec![], start, bump)
                }
            }
            Some(Token::IntSuffix(s))
            | Some(Token::HexIntSuffix(s))
            | Some(Token::BinIntSuffix(s)) => {
                // Copy the bare `&'src str` payload out of the `&Token`
                // binding so the token can be advanced (no `String` is
                // cloned — the payload is a zero-allocation source slice).
                let s: &'src str = *s;
                self.parse_suffixed_int(s, true, start, bump)
            }
            Some(Token::UIntSuffix(s))
            | Some(Token::HexUIntSuffix(s))
            | Some(Token::BinUIntSuffix(s)) => {
                let s: &'src str = *s;
                self.parse_suffixed_int(s, false, start, bump)
            }
            Some(Token::Forall) | Some(Token::Exists) => {
                let quantifier = match self.advance().ok_or_else(|| unreachable!())? {
                    Token::Forall => Quantifier::Forall,
                    Token::Exists => Quantifier::Exists,
                    _ => unreachable!(),
                };
                let binder = match self.advance() {
                    Some(Token::Ident(name)) => name,
                    Some(tok) => {
                        return Err(Diagnostic::error(format!(
                            "expected binder name after {:?}, found {:?}",
                            quantifier, tok
                        ))
                        .with_code_str(ErrCode::E004)
                        .with_help(format!(
                            "`{:?}` must be followed by a binder variable and `in <range>`",
                            quantifier
                        ))
                        .with_suggestion("try `forall i in 0..n: arr[i] > 0`")
                        .with_span(self.span()));
                    }
                    None => {
                        return Err(Diagnostic::error(format!(
                            "unexpected end of file after {:?}",
                            quantifier
                        ))
                        .with_code_str(ErrCode::E002)
                        .with_help("quantified expression is incomplete")
                        .with_span(self.span()));
                    }
                };
                self.expect(Token::In)?;
                let range = self.parse_expr(bump)?;
                self.expect(Token::Colon)?;
                let body = self.parse_expr(bump)?;
                let end = self.span().end;
                Ok(Expr::Quantified {
                    quantifier,
                    binder,
                    range: Self::alloc_shared(bump, range),
                    body: Self::alloc_shared(bump, body),
                    span: Span::new(self.file_id, start, end),
                })
            }
            Some(Token::At) => {
                self.advance();
                let name = match self.advance() {
                    Some(Token::Ident(n)) => n,
                    Some(tok) => {
                        return Err(Diagnostic::error(format!(
                            "expected built-in name after `@`, found {:?}",
                            tok
                        ))
                        .with_span(self.span()));
                    }
                    None => {
                        return Err(Diagnostic::error("unexpected end of input after `@`")
                            .with_span(self.span()));
                    }
                };
                if name.eq_str("typeInfo") {
                    self.expect(Token::Bang)?;
                    self.expect(Token::LParen)?;
                    let ty = self.parse_type(bump)?;
                    self.expect(Token::RParen)?;
                    let end = self.span().end;
                    Ok(Expr::TypeInfo(
                        Self::alloc_shared(bump, ty),
                        Span::new(self.file_id, start, end),
                    ))
                } else if name.eq_str("compile_error") {
                    // The `!` suffix is optional in `@compile_error` per the spec
                    // (the syntax doc omits it but examples use it).
                    if matches!(self.peek(), Some(Token::Bang)) {
                        self.advance();
                    }
                    self.expect(Token::LParen)?;
                    let msg = match self.advance() {
                        Some(Token::StringLiteral(s)) => match parse_string_literal(s) {
                            Ok(c) => c.into_owned(),
                            Err(e) => {
                                return Err(Diagnostic::error(format!(
                                    "@compile_error expects a string literal argument, e.g. `@compile_error(\"message\")` — {}",
                                    e
                                ))
                                .with_span(self.span()))
                            }
                        },
                        _ => return Err(Diagnostic::error("@compile_error expects a string literal argument, e.g. `@compile_error(\"message\")`")
                            .with_span(self.span())),
                    };
                    self.expect(Token::RParen)?;
                    let end = self.span().end;
                    Ok(Expr::CompileError(
                        bump.alloc_str(&msg),
                        Span::new(self.file_id, start, end),
                    ))
                } else {
                    let end = self.span().end;
                    Ok(Expr::Ident(
                        Symbol::intern_prefixed('@', name.as_str()),
                        Span::new(self.file_id, start, end),
                    ))
                }
            }
            _ => {
                let mut diag = Diagnostic::error("expected expression")
                    .with_code_str(ErrCode::E007)
                    .with_span(self.span());
                if let Some(Token::Semicolon) = self.peek() {
                    diag = diag
                        .with_help("remove this extra semicolon")
                        .with_suggestion("remove the `;` here");
                } else {
                    diag = diag.with_help("expected a valid expression — try a literal, variable, `if`, `match`, `|...| { }` closure, or prefix operator")
                        .with_suggestion("try `42`, `true`, `x`, `if cond { a } else { b }`, or `|x| { x + 1 }`");
                }
                Err(diag)
            }
        }
    }

    /// Desugar a suffixed integer literal token (`42i32`, `0xFFu8`) into
    /// `TypeAnnotated { Literal(v), Int|UInt<w> }` — SYNTAX.md §Literals:
    /// "`42i32` (equivalent to `42: Int<32>`) ... The suffix is syntactic
    /// sugar and the type is fully checked."  The desugared `Type` mirrors
    /// the AST `parse_type` builds for a hand-written `Int<8>`
    /// (`Generic(Path([Int]), [Positional(Literal(8))])`).
    ///
    /// The width range is not validated here (the checker owns that, so
    /// the diagnostic spans the width and `42i999` fails exactly like
    /// `42: Int<999>`); an unrepresentable value or width pushes a
    /// diagnostic instead of silently degrading to `Expr::Error`.
    fn parse_suffixed_int(
        &mut self,
        s: &'src str,
        signed: bool,
        start: u32,
        bump: &'input bumpalo::Bump,
    ) -> Result<Expr<'input>, Diagnostic<'src>> {
        self.advance();
        let end = self.span().end;
        let span = Span::new(self.file_id, start, end);
        // Radix from the token's prefix.  The regexes pin lowercase
        // `0x`/`0b` for the hex/bin variants; the uppercase arms are
        // kept so this dispatch matches the old prefix check exactly.
        let radix = if s.starts_with("0x") || s.starts_with("0X") {
            16
        } else if s.starts_with("0b") || s.starts_with("0B") {
            2
        } else {
            10
        };
        // Zero-allocation scan: the old path built a `replace('_', "")`
        // String and then re-scanned it through `from_str_radix`.
        let Some((value, width)) =
            crate::lexer::literal::parse_suffixed_int_literal(bump, s, radix)
        else {
            // The old path returned `Expr::Error` with NO diagnostic, so
            // an overflowing literal silently compiled into an opaque
            // error node.  Mirror `parse_literal`'s overflow diagnostic.
            self.diagnostics.push(
                Diagnostic::error("suffixed integer literal overflow")
                    .with_span(span)
                    .with_help(
                        "the value must fit in `UInt<128>` (the widest \
                         integer type) and the bit width must fit in 32 bits",
                    ),
            );
            return Ok(Expr::Error(span));
        };
        let value_expr = Expr::Literal(Literal::Int(value), span);
        let width_expr =
            Expr::Literal(Literal::Int(crate::ast::IntLit::Small(width as i128)), span);
        let width_ty = Type::Literal(Self::alloc_shared(bump, width_expr), span);
        let base_ty = Type::Path(
            bump.alloc_slice_clone(&[Symbol::intern(if signed { "Int" } else { "UInt" })]),
            span,
        );
        let ty = Type::Generic(
            Self::alloc_shared(bump, base_ty),
            bump.alloc_slice_clone(&[GenericArg::Positional(width_ty)]),
            span,
        );
        let annotated = Expr::TypeAnnotated {
            expr: Self::alloc_shared(bump, value_expr),
            ty: Self::alloc_shared(bump, ty),
            span,
        };
        if matches!(self.peek(), Some(Token::Colon)) {
            // `42i32: T` keeps the old colon-wrap shape: the desugared
            // annotation becomes the inner expression, so the literal
            // must satisfy both `Int<32>` and `T`.
            self.advance();
            let outer_ty = self.parse_type(bump)?;
            let end = self.span().end;
            Ok(Expr::TypeAnnotated {
                expr: Self::alloc_shared(bump, annotated),
                ty: Self::alloc_shared(bump, outer_ty),
                span: Span::new(self.file_id, start, end),
            })
        } else {
            Ok(annotated)
        }
    }

    // -------------------------------------------------------------------------
    // Closure
    // -------------------------------------------------------------------------

    pub(super) fn parse_closure(
        &mut self,
        start: u32,
        bump: &'input bumpalo::Bump,
    ) -> Result<Expr<'input>, Diagnostic<'src>> {
        self.advance();
        let mut params = bumpalo::collections::Vec::with_capacity_in(4, bump);
        loop {
            if matches!(self.peek(), Some(Token::Pipe)) {
                self.advance();
                break;
            }
            let name = match self.advance() {
                Some(Token::Ident(n)) => n,
                Some(tok) => return Err(Diagnostic::error(format!(
                    "expected parameter name, found {:?}",
                    tok
                ))
                .with_code_str(ErrCode::E004)
                .with_help("closure parameters must have a name — e.g. `|x, y| { ... }`")
                .with_suggestion(
                    "use a valid identifier like `x`, `acc`, or `item` for the closure parameter",
                )
                .with_span(self.span())),
                None => return Err(Diagnostic::error("unexpected end of file in closure")
                    .with_code_str(ErrCode::E002)
                    .with_help("closure definition is incomplete — expected `| ... | { ... }`")
                    .with_suggestion(
                        "close the closure with `| { body }` or add parameters: `|x, y| { ... }`",
                    )
                    .with_span(self.span())),
            };
            let ty = if matches!(self.peek(), Some(Token::Colon)) {
                self.advance();
                Some(self.parse_type(bump)?)
            } else {
                None
            };
            params.push(Param {
                name,
                ty,
                default: None,
                span: self.span(),
            });
            if matches!(self.peek(), Some(Token::Comma)) {
                self.advance();
            } else {
                self.expect(Token::Pipe)?;
                break;
            }
        }
        let return_type = if matches!(self.peek(), Some(Token::Arrow)) {
            self.advance();
            Some(self.parse_type(bump)?)
        } else {
            None
        };
        let body = if matches!(self.peek(), Some(Token::LBrace)) {
            self.advance();
            let stmts = self.with_restrictions(ParseRestrictions::VALUE_BLOCK, |this| {
                this.parse_block(bump)
            })?;
            self.expect(Token::RBrace)?;
            stmts
        } else {
            let expr = self.parse_expr(bump)?;
            Self::alloc_slice(bump, &[Stmt::Expression(expr)])
        };
        let end = self.span().end;
        Ok(Expr::Closure {
            params: params.into_bump_slice(),
            return_type,
            captures: Self::alloc_slice(bump, &[]),
            body,
            span: Span::new(self.file_id, start, end),
        })
    }

    // -------------------------------------------------------------------------
    // Path / literal / struct / enum / call
    // -------------------------------------------------------------------------

    fn parse_path_or_literal(
        &mut self,
        start: u32,
        bump: &'input bumpalo::Bump,
    ) -> Result<Expr<'input>, Diagnostic<'src>> {
        let mut path = Vec::new();
        let name = match self.advance() {
            Some(Token::Ident(n)) => n,
            _ => unreachable!(),
        };
        path.push(name);
        while matches!(self.peek(), Some(Token::ColonColon)) {
            self.advance();
            match self.advance() {
                Some(tok) => match tok.as_ident_symbol() {
                    Some(sym) => path.push(sym),
                    None => return Err(Diagnostic::error("expected identifier after '::'")
                        .with_code_str(ErrCode::E004)
                        .with_help(
                            "`::` must be followed by an identifier — e.g. `std::collections::HashMap`",
                        )
                        .with_suggestion("add an identifier after `::`, e.g. `MyModule::MyType`")
                        .with_span(self.span())),
                },
                None => return Err(Diagnostic::error("expected identifier after '::'")
                    .with_code_str(ErrCode::E004)
                    .with_help(
                        "`::` must be followed by an identifier — e.g. `std::collections::HashMap`",
                    )
                    .with_suggestion("add an identifier after `::`, e.g. `MyModule::MyType`")
                    .with_span(self.span())),
            }
        }
        let restrict = self
            .restrictions
            .contains(ParseRestrictions::NO_STRUCT_LITERAL);

        if path.len() == 1 && path[0].eq_str("layout_of") {
            if !matches!(self.peek(), Some(Token::Bang)) {
                let found = match self.peek() {
                    Some(tok) => format!("{:?}", tok),
                    None => "end of file".to_string(),
                };
                return Err(Diagnostic::error(
                    "`layout_of` requires `!` suffix — use `layout_of!(Type)`",
                )
                .with_code_str(ErrCode::E001)
                .with_span(Span::new(self.file_id, start, self.span().end))
                .with_help(format!(
                    "`layout_of` is a comptime intrinsic; add `!` before `(` (found `{}`)",
                    found
                ))
                .with_suggestion("layout_of!(Type)"));
            }
            self.advance();
            self.expect(Token::LParen)?;
            let ty = self.parse_type(bump)?;
            self.expect(Token::RParen)?;
            let end = self.span().end;
            return Ok(Expr::LayoutOf(
                Self::alloc_shared(bump, ty),
                Span::new(self.file_id, start, end),
            ));
        }

        match self.peek() {
            Some(Token::LBrace) if !restrict => self.parse_struct_lit(path, start, bump),
            Some(Token::LParen) => {
                if path.len() == 2 {
                    let variant = path[1];
                    let enum_path = vec![path[0]];
                    self.parse_enum_lit(enum_path, variant, start, bump)
                } else if path.len() >= 2 {
                    let span = Span::new(self.file_id, start, self.span().end);
                    let callee = Expr::Path(bump.alloc_slice_clone(&path), span);
                    self.parse_call(callee, start, bump)
                } else {
                    self.parse_call(
                        Expr::Ident(
                            path.into_iter()
                                .next()
                                .expect("expected at least one path segment"),
                            Span::new(self.file_id, start, self.span().start),
                        ),
                        start,
                        bump,
                    )
                }
            }
            _ => {
                if path.len() >= 2 {
                    let variant = path.pop().expect("Enum pattern must have a variant");
                    let end = self.span().end;
                    Ok(Expr::EnumLit {
                        path: bump.alloc_slice_clone(&path),
                        variant,
                        payload: None,
                        span: Span::new(self.file_id, start, end),
                    })
                } else {
                    Ok(Expr::Ident(
                        path.into_iter()
                            .next()
                            .expect("expected at least one path segment"),
                        Span::new(self.file_id, start, self.span().end),
                    ))
                }
            }
        }
    }

    fn parse_struct_lit(
        &mut self,
        path: Vec<Symbol>,
        start: u32,
        bump: &'input bumpalo::Bump,
    ) -> Result<Expr<'input>, Diagnostic<'src>> {
        self.expect(Token::LBrace)?;
        let mut fields = bumpalo::collections::Vec::with_capacity_in(4, bump);
        loop {
            if matches!(self.peek(), Some(Token::RBrace)) {
                self.advance();
                break;
            }
            let field_name = match self.advance() {
                Some(Token::Ident(n)) => n,
                Some(tok) => return Err(Diagnostic::error(format!(
                    "expected field name, found {:?}",
                    tok
                ))
                .with_code_str(ErrCode::E004)
                .with_help("struct literal fields must have a name — e.g. `Point { x = 1, y = 2 }`")
                .with_suggestion("add a field name followed by `= <value>`, e.g. `field = value`")
                .with_span(self.span())),
                None => {
                    return Err(
                        Diagnostic::error("unexpected end of file in struct literal")
                            .with_code_str(ErrCode::E002)
                            .with_help(
                                "struct literal is incomplete — expected a field name or `}`",
                            )
                            .with_suggestion("close the struct with `}` or add more fields")
                            .with_span(self.span()),
                    );
                }
            };
            self.expect(Token::Assign)?;
            let value = self.parse_expr(bump)?;
            fields.push((field_name, value));
            if matches!(self.peek(), Some(Token::Comma)) {
                self.advance();
            } else {
                self.expect(Token::RBrace)?;
                break;
            }
        }
        let end = self.span().end;
        Ok(Expr::StructLit {
            path: bump.alloc_slice_clone(&path),
            fields: fields.into_bump_slice(),
            span: Span::new(self.file_id, start, end),
        })
    }

    fn parse_enum_lit(
        &mut self,
        path: Vec<Symbol>,
        variant: Symbol,
        start: u32,
        bump: &'input bumpalo::Bump,
    ) -> Result<Expr<'input>, Diagnostic<'src>> {
        self.expect(Token::LParen)?;
        if matches!(self.peek(), Some(Token::RParen)) {
            self.advance();
            let end = self.span().end;
            return Ok(Expr::EnumLit {
                path: bump.alloc_slice_clone(&path),
                variant,
                payload: None,
                span: Span::new(self.file_id, start, end),
            });
        }
        let payload = self.parse_expr(bump)?;
        if matches!(self.peek(), Some(Token::Comma)) {
            let mut args = bumpalo::collections::Vec::with_capacity_in(4, bump);
            args.push(payload);
            while matches!(self.peek(), Some(Token::Comma)) {
                self.advance();
                args.push(self.parse_expr(bump)?);
            }
            self.expect(Token::RParen)?;
            let end = self.span().end;
            let callee = Expr::Path(
                bump.alloc_slice_clone(&path),
                Span::new(self.file_id, start, end),
            );
            return Ok(Expr::Call {
                callee: Self::alloc_shared(bump, callee),
                args: args.into_bump_slice(),
                comptime: false,
                span: Span::new(self.file_id, start, end),
            });
        }
        self.expect(Token::RParen)?;
        let end = self.span().end;
        Ok(Expr::EnumLit {
            path: bump.alloc_slice_clone(&path),
            variant,
            payload: Some(Self::alloc_shared(bump, payload)),
            span: Span::new(self.file_id, start, end),
        })
    }

    fn parse_call(
        &mut self,
        callee: Expr<'input>,
        start: u32,
        bump: &'input bumpalo::Bump,
    ) -> Result<Expr<'input>, Diagnostic<'src>> {
        self.expect(Token::LParen)?;
        let mut args = bumpalo::collections::Vec::with_capacity_in(4, bump);
        if !matches!(self.peek(), Some(Token::RParen)) {
            loop {
                args.push(self.parse_expr(bump)?);
                if matches!(self.peek(), Some(Token::Comma)) {
                    self.advance();
                } else {
                    break;
                }
            }
        }
        self.expect(Token::RParen)?;
        let end = self.span().end;
        Ok(Expr::Call {
            callee: Self::alloc_shared(bump, callee),
            args: args.into_bump_slice(),
            comptime: false,
            span: Span::new(self.file_id, start, end),
        })
    }

    // -------------------------------------------------------------------------
    // Literal parsing
    // -------------------------------------------------------------------------

    pub(super) fn parse_literal(
        &mut self,
        bump: &'input bumpalo::Bump,
    ) -> Result<Expr<'input>, Diagnostic<'src>> {
        let start = self.span().start;
        let token = self.advance().ok_or_else(|| {
            Diagnostic::error("unexpected token")
                .with_code_str(ErrCode::E003)
                .with_help("expected a literal value (number, string, char, bool, or byte string)")
                .with_span(Span::new(self.file_id, 0, 0))
        })?;
        let end = self.span().end;
        let span = Span::new(self.file_id, start, end);
        // The token carries the raw source slice; parsing happens HERE, at
        // consumption time.  A malformed literal surfaces exactly where (and
        // only where) it is consumed: a diagnostic carrying the same message
        // the lex-time callback used to store in the token's `Err` payload,
        // anchored on the literal's own span.
        match token {
            Token::IntLiteral(s) => {
                match parse_int_literal(bump, s, 10, "integer literal overflow") {
                    Ok(v) => Ok(Expr::Literal(Literal::Int(v), span)),
                    Err(e) => {
                        self.diagnostics
                            .push(Diagnostic::error(e.as_str()).with_span(span));
                        Ok(Expr::Error(span))
                    }
                }
            }
            Token::HexLiteral(s) => {
                match parse_int_literal(bump, &s[2..], 16, "hex literal overflow") {
                    Ok(v) => Ok(Expr::Literal(Literal::Int(v), span)),
                    Err(e) => {
                        self.diagnostics
                            .push(Diagnostic::error(e.as_str()).with_span(span));
                        Ok(Expr::Error(span))
                    }
                }
            }
            Token::BinLiteral(s) => {
                match parse_int_literal(bump, &s[2..], 2, "binary literal overflow") {
                    Ok(v) => Ok(Expr::Literal(Literal::Int(v), span)),
                    Err(e) => {
                        self.diagnostics
                            .push(Diagnostic::error(e.as_str()).with_span(span));
                        Ok(Expr::Error(span))
                    }
                }
            }
            Token::FloatLiteral(s) => match parse_float_literal(s) {
                Ok(v) => Ok(Expr::Literal(Literal::Float(v), span)),
                // Float default `trap` (aligned with integers): a
                // compile-time float-literal anomaly
                // (overflow/NaN/inf) is a COMPILE ERROR — the error must
                // not be swallowed into Expr::Error.
                Err(e) => {
                    return Err(Diagnostic::error(e.as_str()).with_span(span));
                }
            },
            Token::CharLiteral(s) => match parse_char_literal(s) {
                Ok(v) => Ok(Expr::Literal(Literal::Char(v), span)),
                Err(e) => {
                    self.diagnostics
                        .push(Diagnostic::error(e.as_str()).with_span(span));
                    Ok(Expr::Error(span))
                }
            },
            Token::ByteCharLiteral(s) => match parse_byte_char_literal(s) {
                Ok(v) => Ok(Expr::Literal(Literal::Byte(v), span)),
                Err(e) => {
                    self.diagnostics
                        .push(Diagnostic::error(e.as_str()).with_span(span));
                    Ok(Expr::Error(span))
                }
            },
            Token::StringLiteral(s) => match parse_string_literal(s) {
                Ok(c) => {
                    // Arena-copy the decoded text: `Literal::String` holds
                    // `&'input str`, and borrowing the token slice directly
                    // would tie the AST to the source-text lifetime, which
                    // the arena is not guaranteed to outlive.  The escape
                    // path already produced an owned `String`, so only the
                    // common escape-free case pays a small copy.
                    let s: &'input str = bump.alloc_str(c.as_ref());
                    Ok(Expr::Literal(Literal::String(s), span))
                }
                Err(e) => {
                    self.diagnostics
                        .push(Diagnostic::error(e.as_str()).with_span(span));
                    Ok(Expr::Error(span))
                }
            },
            Token::ByteStringLiteral(s) => match parse_byte_string_literal(s) {
                Ok(c) => {
                    // Always arena-copy: the `Borrowed` case borrows the
                    // token slice with the source lifetime, which the AST
                    // arena is not guaranteed to outlive.
                    let b: &'input [u8] = bump.alloc_slice_clone(c.as_ref());
                    Ok(Expr::Literal(Literal::ByteString(b), span))
                }
                Err(e) => {
                    self.diagnostics
                        .push(Diagnostic::error(e.as_str()).with_span(span));
                    Ok(Expr::Error(span))
                }
            },
            Token::True => Ok(Expr::Literal(Literal::Bool(true), span)),
            Token::False => Ok(Expr::Literal(Literal::Bool(false), span)),
            _ => unreachable!(),
        }
    }

    // -------------------------------------------------------------------------
    // Infix parsing
    // -------------------------------------------------------------------------

    fn parse_infix(
        &mut self,
        lhs: Expr<'input>,
        bp: u8,
        bump: &'input bumpalo::Bump,
    ) -> Result<Expr<'input>, Diagnostic<'src>> {
        let start = self.span().start;
        match self.peek() {
            Some(Token::Bang) => {
                if matches!(lhs, Expr::Ident(..) | Expr::Path(..))
                    && matches!(self.peek_next(), Some(Token::LParen))
                {
                    self.advance();
                    self.advance();
                    let mut args = bumpalo::collections::Vec::with_capacity_in(4, bump);
                    if !matches!(self.peek(), Some(Token::RParen)) {
                        loop {
                            args.push(self.parse_expr(bump)?);
                            if matches!(self.peek(), Some(Token::Comma)) {
                                self.advance();
                            } else {
                                break;
                            }
                        }
                    }
                    self.expect(Token::RParen)?;
                    Ok(Expr::Call {
                        callee: Self::alloc_shared(bump, lhs),
                        args: args.into_bump_slice(),
                        comptime: true,
                        span: Span::new(self.file_id, start, self.span().end),
                    })
                } else {
                    Err(Diagnostic::error("unexpected !")
                        .with_code_str(ErrCode::E007)
                        .with_help("`!` as a postfix operator is only used for comptime calls — e.g. `func!()`")
                        .with_suggestion("use `func!()` for comptime calls, or remove the `!`")
                        .with_span(self.span()))
                }
            }
            Some(Token::Question) => {
                self.advance();
                let end = self.span().end;
                Ok(Expr::Try {
                    expr: Self::alloc_shared(bump, lhs),
                    span: Span::new(self.file_id, start, end),
                })
            }
            Some(Token::As) => {
                self.advance();
                let safe = !matches!(self.peek(), Some(Token::Bang));
                if !safe {
                    self.advance();
                }
                let ty = self.parse_type(bump)?;
                // `round` / `trunc` / `ceil` / `floor` are contextual
                // keywords (not in SYNTAX.md's keyword list) — matched on
                // the identifier, never globally reserved.
                let rounding = match self.peek() {
                    Some(Token::Ident(s)) if s.eq_str("round") => {
                        self.advance();
                        Some(Rounding::Round)
                    }
                    Some(Token::Ident(s)) if s.eq_str("trunc") => {
                        self.advance();
                        Some(Rounding::Trunc)
                    }
                    Some(Token::Ident(s)) if s.eq_str("ceil") => {
                        self.advance();
                        Some(Rounding::Ceil)
                    }
                    Some(Token::Ident(s)) if s.eq_str("floor") => {
                        self.advance();
                        Some(Rounding::Floor)
                    }
                    _ => None,
                };
                let end = self.span().end;
                Ok(Expr::Cast {
                    expr: Self::alloc_shared(bump, lhs),
                    ty: Self::alloc_shared(bump, ty),
                    safe,
                    rounding,
                    span: Span::new(self.file_id, start, end),
                })
            }
            Some(Token::Plus) => {
                self.advance();
                self.binary(lhs, BinOp::Add, bp + 1, start, bump)
            }
            Some(Token::Minus) => {
                self.advance();
                self.binary(lhs, BinOp::Sub, bp + 1, start, bump)
            }
            Some(Token::Star) => {
                self.advance();
                self.binary(lhs, BinOp::Mul, bp + 1, start, bump)
            }
            Some(Token::Slash) => {
                self.advance();
                self.binary(lhs, BinOp::Div, bp + 1, start, bump)
            }
            Some(Token::Percent) => {
                self.advance();
                self.binary(lhs, BinOp::Rem, bp + 1, start, bump)
            }
            Some(Token::PlusWrap) => {
                self.advance();
                self.binary(lhs, BinOp::AddWrap, bp + 1, start, bump)
            }
            Some(Token::MinusWrap) => {
                self.advance();
                self.binary(lhs, BinOp::SubWrap, bp + 1, start, bump)
            }
            Some(Token::StarWrap) => {
                self.advance();
                self.binary(lhs, BinOp::MulWrap, bp + 1, start, bump)
            }
            Some(Token::PlusSaturate) => {
                self.advance();
                self.binary(lhs, BinOp::AddSaturate, bp + 1, start, bump)
            }
            Some(Token::MinusSaturate) => {
                self.advance();
                self.binary(lhs, BinOp::SubSaturate, bp + 1, start, bump)
            }
            Some(Token::StarSaturate) => {
                self.advance();
                self.binary(lhs, BinOp::MulSaturate, bp + 1, start, bump)
            }
            Some(Token::PlusTrap) => {
                self.advance();
                self.binary(lhs, BinOp::AddTrap, bp + 1, start, bump)
            }
            Some(Token::MinusTrap) => {
                self.advance();
                self.binary(lhs, BinOp::SubTrap, bp + 1, start, bump)
            }
            Some(Token::StarTrap) => {
                self.advance();
                self.binary(lhs, BinOp::MulTrap, bp + 1, start, bump)
            }
            Some(Token::Ampersand) => {
                self.advance();
                self.binary(lhs, BinOp::BitAnd, bp + 1, start, bump)
            }
            Some(Token::Pipe) => {
                self.advance();
                self.binary(lhs, BinOp::BitOr, bp + 1, start, bump)
            }
            Some(Token::Caret) => {
                self.advance();
                self.binary(lhs, BinOp::BitXor, bp + 1, start, bump)
            }
            Some(Token::Shl) => {
                self.advance();
                self.binary(lhs, BinOp::Shl, bp + 1, start, bump)
            }
            Some(Token::Shr) => {
                self.advance();
                self.binary(lhs, BinOp::Shr, bp + 1, start, bump)
            }
            Some(Token::EqEq) => {
                self.advance();
                self.binary(lhs, BinOp::Eq, bp + 1, start, bump)
            }
            Some(Token::Neq) => {
                self.advance();
                self.binary(lhs, BinOp::Neq, bp + 1, start, bump)
            }
            Some(Token::Lt) => {
                self.advance();
                self.binary(lhs, BinOp::Lt, bp + 1, start, bump)
            }
            Some(Token::Gt) => {
                self.advance();
                self.binary(lhs, BinOp::Gt, bp + 1, start, bump)
            }
            Some(Token::Le) => {
                self.advance();
                self.binary(lhs, BinOp::Le, bp + 1, start, bump)
            }
            Some(Token::Ge) => {
                self.advance();
                self.binary(lhs, BinOp::Ge, bp + 1, start, bump)
            }
            Some(Token::And) => {
                self.advance();
                self.binary(lhs, BinOp::And, bp + 1, start, bump)
            }
            Some(Token::Or) => {
                self.advance();
                self.binary(lhs, BinOp::Or, bp + 1, start, bump)
            }
            Some(Token::DotDot) => {
                self.advance();
                let end = if !matches!(
                    self.peek(),
                    Some(Token::Semicolon)
                        | Some(Token::RBrace)
                        | Some(Token::Comma)
                        | Some(Token::RParen)
                ) {
                    Some(Self::alloc_shared(bump, self.parse_expr_bp(0, bump)?))
                } else {
                    None
                };
                Ok(Expr::Range {
                    start: Some(Self::alloc_shared(bump, lhs)),
                    end,
                    inclusive: false,
                    span: Span::new(self.file_id, start, self.span().end),
                })
            }
            Some(Token::DotDotEq) => {
                self.advance();
                let end = self.parse_expr_bp(0, bump)?;
                Ok(Expr::Range {
                    start: Some(Self::alloc_shared(bump, lhs)),
                    end: Some(Self::alloc_shared(bump, end)),
                    inclusive: true,
                    span: Span::new(self.file_id, start, self.span().end),
                })
            }
            Some(Token::LParen) => {
                self.advance();
                let mut args = bumpalo::collections::Vec::with_capacity_in(4, bump);
                if !matches!(self.peek(), Some(Token::RParen)) {
                    loop {
                        args.push(self.parse_expr(bump)?);
                        if matches!(self.peek(), Some(Token::Comma)) {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                }
                self.expect(Token::RParen)?;
                Ok(Expr::Call {
                    callee: Self::alloc_shared(bump, lhs),
                    args: args.into_bump_slice(),
                    comptime: false,
                    span: Span::new(self.file_id, start, self.span().end),
                })
            }
            Some(Token::LBracket) => {
                self.advance();
                let index = self.parse_expr(bump)?;
                self.expect(Token::RBracket)?;
                Ok(Expr::Index {
                    base: Self::alloc_shared(bump, lhs),
                    index: Self::alloc_shared(bump, index),
                    span: Span::new(self.file_id, start, self.span().end),
                })
            }
            Some(Token::Dot) => {
                self.advance();
                if let Some(Token::Ident(field)) = self.advance() {
                    if field.eq_str("freeze") && matches!(self.peek(), Some(Token::Bang)) {
                        // `.freeze!()` — explicit read-only freeze, equivalent
                        // to `&ro expr` (SYNTAX.md: ".freeze!() ... behaves
                        // identically and is preferred in method chains").
                        self.advance();
                        self.expect(Token::LParen)?;
                        self.expect(Token::RParen)?;
                        Ok(Expr::UnaryOp {
                            op: crate::ast::UnaryOp::Ro,
                            expr: Self::alloc_shared(bump, lhs),
                            span: Span::new(self.file_id, start, self.span().end),
                        })
                    } else {
                        Ok(Expr::FieldAccess {
                            base: Self::alloc_shared(bump, lhs),
                            field,
                            span: Span::new(self.file_id, start, self.span().end),
                        })
                    }
                } else {
                    Err(Diagnostic::error("expected field name after '.'")
                        .with_code_str(ErrCode::E004)
                        .with_help("`.` must be followed by a field name — e.g. `object.field`")
                        .with_suggestion("add a field name after `.`, or remove the `.`")
                        .with_span(self.span()))
                }
            }
            Some(Token::Apostrophe) => {
                self.advance();
                if let Some(Token::Ident(attr)) = self.advance() {
                    Ok(Expr::AttrAccess {
                        base: Self::alloc_shared(bump, lhs),
                        attr,
                        span: Span::new(self.file_id, start, self.span().end),
                    })
                } else {
                    Err(Diagnostic::error("expected attribute name after '''")
                        .with_code_str(ErrCode::E004)
                        .with_help("`'` must be followed by an attribute name — e.g. `object'attr`")
                        .with_suggestion("add an attribute name after `'`, or remove the `'`")
                        .with_span(self.span()))
                }
            }
            Some(Token::Catch) => {
                self.advance();
                let mut branches = bumpalo::collections::Vec::with_capacity_in(4, bump);
                self.expect(Token::LBrace)?;
                loop {
                    if matches!(self.peek(), Some(Token::RBrace)) {
                        self.advance();
                        break;
                    }
                    let branch_start = self.span().start;
                    self.expect(Token::Pipe)?;
                    let pattern = self.parse_pattern(bump)?;
                    let bind = if matches!(self.peek(), Some(Token::As)) {
                        self.advance();
                        match self.advance() {
                            Some(Token::Ident(name)) => Some(name),
                            _ => return Err(Diagnostic::error("expected binding name after 'as'")
                                .with_code_str(ErrCode::E004)
                                .with_help("`as` in a catch pattern must be followed by a binding name — e.g. `|NetworkError as e|`")
                                .with_suggestion("add a capture variable name after `as`, like `|Pattern as var_name|`")
                                .with_span(self.span())),
                        }
                    } else {
                        None
                    };
                    self.expect(Token::Pipe)?;
                    let body = if matches!(self.peek(), Some(Token::FatArrow)) {
                        self.advance();
                        let expr = self.parse_expr(bump)?;
                        Self::alloc_slice(bump, &[Stmt::Expression(expr)])
                    } else {
                        self.expect(Token::LBrace)?;
                        let block = self
                            .with_restrictions(ParseRestrictions::VALUE_BLOCK, |this| {
                                this.parse_block(bump)
                            })?;
                        self.expect(Token::RBrace)?;
                        block
                    };
                    branches.push(CatchBranch {
                        pattern,
                        bind,
                        body,
                        span: Span::new(self.file_id, branch_start, self.span().end),
                    });
                }
                Ok(Expr::Catch {
                    expr: Self::alloc_shared(bump, lhs),
                    branches: branches.into_bump_slice(),
                    span: Span::new(self.file_id, start, self.span().end),
                })
            }
            _ => unreachable!(),
        }
    }

    fn binary(
        &mut self,
        lhs: Expr<'input>,
        op: BinOp,
        bp: u8,
        start: u32,
        bump: &'input bumpalo::Bump,
    ) -> Result<Expr<'input>, Diagnostic<'src>> {
        let rhs = self.parse_expr_bp(bp, bump)?;
        Ok(Expr::BinaryOp {
            left: Self::alloc_shared(bump, lhs),
            op,
            right: Self::alloc_shared(bump, rhs),
            span: Span::new(self.file_id, start, self.span().end),
        })
    }

    // -------------------------------------------------------------------------
    // `if` expression
    // -------------------------------------------------------------------------

    fn parse_if_expr(
        &mut self,
        bump: &'input bumpalo::Bump,
    ) -> Result<Expr<'input>, Diagnostic<'src>> {
        let start = self.span().start;
        self.advance();
        if matches!(self.peek(), Some(Token::Let)) {
            self.advance();
            let pattern = self.parse_or_pattern(bump)?;
            self.expect(Token::Assign)?;
            let scrutinee = self
                .with_restrictions(ParseRestrictions::NO_STRUCT_LITERAL, |this| {
                    this.parse_expr(bump)
                })?;
            self.expect(Token::LBrace)?;
            let then_branch = self.with_restrictions(ParseRestrictions::VALUE_BLOCK, |this| {
                this.parse_block(bump)
            })?;
            self.expect(Token::RBrace)?;
            let else_branch = if matches!(self.peek(), Some(Token::Else)) {
                self.advance();
                if matches!(self.peek(), Some(Token::If)) {
                    Some(Self::alloc_slice(
                        bump,
                        &[Stmt::Expression(self.parse_if_expr(bump)?)],
                    ))
                } else {
                    self.expect(Token::LBrace)?;
                    let block = self.with_restrictions(ParseRestrictions::VALUE_BLOCK, |this| {
                        this.parse_block(bump)
                    })?;
                    self.expect(Token::RBrace)?;
                    Some(block)
                }
            } else {
                None
            };
            let end = self.span().end;
            return Ok(Expr::IfLet {
                pattern,
                scrutinee: Self::alloc_shared(bump, scrutinee),
                then_branch,
                else_branch,
                is_expression: true,
                span: Span::new(self.file_id, start, end),
            });
        }
        let cond = self.with_restrictions(ParseRestrictions::NO_STRUCT_LITERAL, |this| {
            this.parse_expr(bump)
        })?;
        self.expect(Token::LBrace)?;
        let then_branch = self.with_restrictions(ParseRestrictions::VALUE_BLOCK, |this| {
            this.parse_block(bump)
        })?;
        self.expect(Token::RBrace)?;
        let else_branch = if matches!(self.peek(), Some(Token::Else)) {
            self.advance();
            if matches!(self.peek(), Some(Token::If)) {
                Some(Self::alloc_slice(
                    bump,
                    &[Stmt::Expression(self.parse_if_expr(bump)?)],
                ))
            } else {
                self.expect(Token::LBrace)?;
                let block = self.with_restrictions(ParseRestrictions::VALUE_BLOCK, |this| {
                    this.parse_block(bump)
                })?;
                self.expect(Token::RBrace)?;
                Some(block)
            }
        } else {
            None
        };
        let end = self.span().end;
        Ok(Expr::If {
            cond: Self::alloc_shared(bump, cond),
            then_branch,
            else_branch,
            is_expression: true,
            span: Span::new(self.file_id, start, end),
        })
    }

    // -------------------------------------------------------------------------
    // Match expression
    // -------------------------------------------------------------------------

    pub(super) fn parse_match_expr(
        &mut self,
        bump: &'input bumpalo::Bump,
    ) -> Result<Expr<'input>, Diagnostic<'src>> {
        let start = self.span().start;
        self.advance();
        let scrutinee = self.with_restrictions(ParseRestrictions::NO_STRUCT_LITERAL, |this| {
            this.parse_expr(bump)
        })?;
        self.expect(Token::LBrace)?;
        let mut arms = bumpalo::collections::Vec::with_capacity_in(4, bump);
        loop {
            if matches!(self.peek(), Some(Token::RBrace)) {
                self.advance();
                break;
            }
            let arm_start = self.span().start;
            let pattern = self.parse_or_pattern(bump)?;
            let guard = if matches!(self.peek(), Some(Token::If)) {
                self.advance();
                Some(Self::alloc_shared(bump, self.parse_expr(bump)?))
            } else {
                None
            };
            self.expect(Token::FatArrow)?;
            let body = self.with_restrictions(
                ParseRestrictions::VALUE_BLOCK | ParseRestrictions::NO_STRUCT_LITERAL,
                |this| this.parse_expr(bump),
            )?;
            arms.push(MatchArm {
                pattern,
                guard,
                body,
                span: Span::new(self.file_id, arm_start, self.span().end),
            });
            if matches!(self.peek(), Some(Token::Comma)) {
                self.advance();
            }
            if matches!(self.peek(), Some(Token::RBrace)) {
                self.advance();
                break;
            }
        }
        let end = self.span().end;
        Ok(Expr::Match {
            scrutinee: Self::alloc_shared(bump, scrutinee),
            arms: arms.into_bump_slice(),
            span: Span::new(self.file_id, start, end),
        })
    }
}
