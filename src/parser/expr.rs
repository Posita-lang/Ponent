//! Pratt‑style expression parser with prefix/infix handling and all expression constructs.

use super::*;
use crate::ast::*;
use crate::diagnostics::Diagnostic;
use crate::lexer::Token;
use crate::symbol::Symbol;
use smallvec::smallvec;

impl<'input> Parser<'input> {
    // -------------------------------------------------------------------------
    // Expression entry
    // -------------------------------------------------------------------------

    pub fn parse_expr(&mut self) -> Result<Expr<'input>, Diagnostic> {
        if let Some(e) = self.parse_expr_fast()? {
            return Ok(e);
        }
        self.parse_expr_bp(0)
    }

    /// Fast path for plain literals followed by a terminator.
    /// Hot path (the industrial-engineering pass): the overwhelming
    /// majority of expressions are plain literals — build them directly
    /// without the full Pratt machinery.  A following `:` (type
    /// annotation) or anything else falls back to the general parser.
    #[inline(always)]
    fn parse_expr_fast(&mut self) -> Result<Option<Expr<'input>>, Diagnostic> {
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
            Ok(Token::IntLiteral(_))
            | Ok(Token::HexLiteral(_))
            | Ok(Token::BinLiteral(_))
            | Ok(Token::FloatLiteral(_))
            | Ok(Token::StringLiteral(_))
            | Ok(Token::ByteStringLiteral(_))
            | Ok(Token::CharLiteral(_))
            | Ok(Token::True)
            | Ok(Token::False) => {
                let expr = self.parse_literal()?;
                Ok(Some(expr))
            }
            _ => Ok(None),
        }
    }

    // -------------------------------------------------------------------------
    // Pratt parser (binding power)
    // -------------------------------------------------------------------------

    fn parse_expr_bp(&mut self, min_bp: u8) -> Result<Expr<'input>, Diagnostic> {
        self.recursion_depth += 1;
        if self.recursion_depth > self.max_recursion_depth {
            self.recursion_depth -= 1;
            return Err(Diagnostic::error(format!("maximum recursion depth {} exceeded", self.max_recursion_depth))
                .with_code_str("E006")
                .with_help("the parser reached its recursion limit — the type/structure may be deeply nested or self-referential")
                .with_suggestion("try breaking up deeply nested structures, or use less complex type expressions")
                .with_span(self.span()));
        }
        let result = self.parse_expr_bp_inner(min_bp);
        self.recursion_depth -= 1;
        result
    }

    fn parse_expr_bp_inner(&mut self, min_bp: u8) -> Result<Expr<'input>, Diagnostic> {
        let mut lhs = self.parse_prefix()?;
        loop {
            let token_opt = self.peek().as_ref().ok().cloned();
            if matches!(token_opt, None)
                || matches!(
                    token_opt,
                    Some(Token::Semicolon)
                        | Some(Token::RBrace)
                        | Some(Token::RParen)
                        | Some(Token::Comma)
                        | Some(Token::Colon)
                        | Some(Token::In)
                )
            {
                break;
            }
            if let Some((lbp, _)) = self.prefix_binding_power(token_opt.as_ref()) {
                if lbp < min_bp {
                    break;
                }
                if self.restrictions.contains(ParseRestrictions::NO_COMPARISON) {
                    let is_compare = token_opt.is_some_and(|t| {
                        crate::lexer::token_class(&t).contains(crate::lexer::TokenClass::COMPARISON)
                    });
                    if is_compare {
                        break;
                    }
                }
                lhs = self.parse_infix(lhs, lbp)?;
            } else {
                break;
            }
        }
        Ok(lhs)
    }

    fn prefix_binding_power(&self, token: Option<&Token>) -> Option<(u8, bool)> {
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

    fn parse_prefix(&mut self) -> Result<Expr<'input>, Diagnostic> {
        let start = self.span().start;
        match self.peek() {
            Ok(Token::IntLiteral(_))
            | Ok(Token::HexLiteral(_))
            | Ok(Token::BinLiteral(_))
            | Ok(Token::FloatLiteral(_))
            | Ok(Token::StringLiteral(_))
            | Ok(Token::ByteStringLiteral(_))
            | Ok(Token::CharLiteral(_))
            | Ok(Token::True)
            | Ok(Token::False) => {
                let expr = self.parse_literal()?;
                if matches!(self.peek(), Ok(Token::Colon)) {
                    self.advance().ok();
                    let ty = self.parse_type()?;
                    let end = self.span().end;
                    Ok(Expr::TypeAnnotated {
                        expr: Self::alloc_shared(self.arena, expr),
                        ty: Self::alloc_shared(self.arena, ty),
                        span: Span::new(start, end),
                    })
                } else {
                    Ok(expr)
                }
            }
            Ok(Token::Ident(_)) => self.parse_path_or_literal(start),
            Ok(Token::LParen) => {
                self.advance().ok();
                if matches!(self.peek(), Ok(Token::RParen)) {
                    self.advance().ok();
                    let end = self.span().end;
                    return Ok(Expr::Tuple(Vec::new(), Span::new(start, end)));
                }
                let expr = self.parse_expr()?;
                if matches!(self.peek(), Ok(Token::Comma)) {
                    let mut exprs = vec![expr];
                    while matches!(self.peek(), Ok(Token::Comma)) {
                        self.advance().ok();
                        exprs.push(self.parse_expr()?);
                    }
                    self.expect(Token::RParen)?;
                    let end = self.span().end;
                    Ok(Expr::Tuple(exprs, Span::new(start, end)))
                } else {
                    self.expect(Token::RParen)?;
                    Ok(expr)
                }
            }
            Ok(Token::LBracket) => {
                self.advance().ok();
                let mut exprs = Vec::new();
                loop {
                    if matches!(self.peek(), Ok(Token::RBracket)) {
                        self.advance().ok();
                        break;
                    }
                    exprs.push(self.parse_expr()?);
                    if matches!(self.peek(), Ok(Token::Comma)) {
                        self.advance().ok();
                    } else {
                        self.expect(Token::RBracket)?;
                        break;
                    }
                }
                let end = self.span().end;
                Ok(Expr::Array(exprs, Span::new(start, end)))
            }
            Ok(Token::Plus) | Ok(Token::Minus) | Ok(Token::Star) | Ok(Token::Slash)
            | Ok(Token::Percent) => {
                let next = self.peek_next();
                let is_operator_arg = matches!(
                    next,
                    Some(Token::Comma)
                        | Some(Token::RParen)
                        | Some(Token::RBracket)
                        | Some(Token::RBrace)
                );
                if is_operator_arg {
                    let op_tok = self.advance().map_err(|_| {
                        Diagnostic::error("unexpected token")
                            .with_code_str("E003")
                            .with_help("expected an expression after the operator position")
                            .with_span(Span::new(0, 0))
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
                    Ok(Expr::Ident(op_name, Span::new(start, end)))
                } else {
                    match self.advance().map_err(|_| Diagnostic::error("unexpected token")
                        .with_code_str("E003")
                        .with_help("expected a unary operator (`-`, `*`, `!`, `~`, `&`, `move`)")
                        .with_span(Span::new(0, 0)))?
                    {
                        Token::Minus => {
                            let expr = self.parse_prefix()?;
                            let end = self.span().end;
                            Ok(Expr::UnaryOp { op: UnaryOp::Neg, expr: Self::alloc_shared(self.arena, expr), span: Span::new(start, end) })
                        }
                        Token::Star => {
                            let expr = self.parse_prefix()?;
                            let end = self.span().end;
                            Ok(Expr::UnaryOp { op: UnaryOp::Deref, expr: Self::alloc_shared(self.arena, expr), span: Span::new(start, end) })
                        }
                        _ => Err(Diagnostic::error("unexpected operator in expression")
                            .with_code_str("E007")
                            .with_help("this operator is not valid at this position — check for missing operands or extra operators")
                            .with_span(self.span())),
                    }
                }
            }
            Ok(Token::If) => self.parse_if_expr(),
            Ok(Token::Task) => {
                let start = self.span().start;
                self.advance().ok();
                self.expect(Token::LBrace)?;
                let body = self.parse_block()?;
                self.expect(Token::RBrace)?;
                let end = self.span().end;
                Ok(Expr::Task {
                    body,
                    span: Span::new(start, end),
                })
            }
            Ok(Token::Match) => self.parse_match_expr(),
            Ok(Token::Leave) => {
                self.advance().ok();
                self.expect(Token::With)?;
                let expr = self.parse_expr()?;
                let end = self.span().end;
                Ok(Expr::LeaveWith {
                    expr: Self::alloc_shared(self.arena, expr),
                    is_return: false,
                    span: Span::new(start, end),
                })
            }
            Ok(Token::Return) => {
                // Consume the `return` keyword FIRST — the value
                // expression below is parsed from the NEXT token;
                // otherwise `parse_expr` on the value re-enters this
                // branch with the unconsumed `return` (an infinite
                // recursion — stack overflow — exposed by expression-
                // position returns such as a match arm's `return 0`).
                self.advance().ok();
                let value = if !matches!(self.peek(), Ok(Token::Semicolon) | Ok(Token::RBrace)) {
                    Some(Self::alloc_shared(self.arena, self.parse_expr()?))
                } else {
                    None
                };
                let end = self.span().end;
                Ok(Expr::LeaveWith {
                    expr: value.unwrap_or_else(|| {
                        Self::alloc_shared(
                            self.arena,
                            Expr::Tuple(Vec::new(), Span::new(start, end)),
                        )
                    }),
                    is_return: true,
                    span: Span::new(start, end),
                })
            }
            Ok(Token::Await) => {
                self.advance().ok();
                let expr = self.parse_expr()?;
                let end = self.span().end;
                Ok(Expr::Await {
                    expr: Self::alloc_shared(self.arena, expr),
                    span: Span::new(start, end),
                })
            }
            Ok(Token::Poly) => {
                self.advance().ok();
                self.expect(Token::LParen)?;
                let expr = self.parse_expr()?;
                self.expect(Token::RParen)?;
                let end = self.span().end;
                Ok(Expr::PolyBox {
                    expr: Self::alloc_shared(self.arena, expr),
                    scheme: None,
                    span: Span::new(start, end),
                })
            }
            Ok(Token::Unbox) => {
                self.advance().ok();
                self.expect(Token::LParen)?;
                let expr = self.parse_expr()?;
                self.expect(Token::RParen)?;
                let end = self.span().end;
                Ok(Expr::PolyUnbox {
                    expr: Self::alloc_shared(self.arena, expr),
                    scheme: None,
                    span: Span::new(start, end),
                })
            }
            Ok(Token::Old) => {
                self.advance().ok();
                self.expect(Token::LParen)?;
                let expr = self.parse_expr()?;
                self.expect(Token::RParen)?;
                let end = self.span().end;
                Ok(Expr::Old(
                    Self::alloc_shared(self.arena, expr),
                    Span::new(start, end),
                ))
            }
            Ok(Token::Not) => {
                self.advance().ok();
                let expr = self.parse_expr_bp(1)?;
                let end = self.span().end;
                Ok(Expr::UnaryOp {
                    op: UnaryOp::Not,
                    expr: Self::alloc_shared(self.arena, expr),
                    span: Span::new(start, end),
                })
            }
            Ok(Token::Tilde) => {
                self.advance().ok();
                let expr = self.parse_prefix()?;
                let end = self.span().end;
                Ok(Expr::UnaryOp {
                    op: UnaryOp::BitNot,
                    expr: Self::alloc_shared(self.arena, expr),
                    span: Span::new(start, end),
                })
            }
            Ok(Token::Ampersand) => {
                self.advance().ok();
                // `&ro` freezes a `&mut T` into a `&T` (SYNTAX.md §Reference
                // Coercion): `ro` only has special meaning immediately
                // following `&` in a borrow expression.
                let read_only = matches!(self.peek(), Ok(Token::Ident(s)) if s.eq_str("ro"));
                if read_only {
                    self.advance().ok();
                }
                let mutable = !read_only && matches!(self.peek(), Ok(Token::Mut));
                if mutable {
                    self.advance().ok();
                }
                // The borrow's operand must include the POSTFIX chain
                // (`&mut arr[0]` = `&mut (arr[0])`, NOT `(&mut arr)[0]`):
                // parse at binding power 18 (= the postfix operators'
                // binding power, above every binary operator) so the
                // borrow wraps the whole index/field/call — mirroring
                // rustc's `parse_expr_borrow` → `parse_expr_prefix`.
                let expr = self.parse_expr_bp(18)?;
                let end = self.span().end;
                Ok(Expr::UnaryOp {
                    op: if read_only {
                        UnaryOp::Ro
                    } else if mutable {
                        UnaryOp::RefMut
                    } else {
                        UnaryOp::Ref
                    },
                    expr: Self::alloc_shared(self.arena, expr),
                    span: Span::new(start, end),
                })
            }
            Ok(Token::Move) => {
                self.advance().ok();
                let expr = self.parse_prefix()?;
                let end = self.span().end;
                Ok(Expr::Move(
                    Self::alloc_shared(self.arena, expr),
                    Span::new(start, end),
                ))
            }
            Ok(Token::Unsafe) => {
                self.advance().ok();
                self.expect(Token::LBrace)?;
                let body = self.parse_block()?;
                self.expect(Token::RBrace)?;
                let end = self.span().end;
                Ok(Expr::UnsafeBlock {
                    body,
                    span: Span::new(start, end),
                })
            }
            Ok(Token::Pipe) => self.parse_closure(start),
            Ok(Token::LBrace) => {
                if self
                    .restrictions
                    .contains(ParseRestrictions::NO_STRUCT_LITERAL)
                {
                    self.advance().ok();
                    let body = self.parse_block()?;
                    self.expect(Token::RBrace)?;
                    Ok(Expr::Block(body, Span::new(start, self.span().end)))
                } else {
                    self.parse_struct_lit(vec![], start)
                }
            }
            Ok(Token::IntSuffix(s))
            | Ok(Token::UIntSuffix(s))
            | Ok(Token::HexIntSuffix(s))
            | Ok(Token::HexUIntSuffix(s))
            | Ok(Token::BinIntSuffix(s))
            | Ok(Token::BinUIntSuffix(s)) => {
                let s = s.clone();
                self.advance().ok();
                let end = self.span().end;
                let value = if s.starts_with("0x") || s.starts_with("0X") {
                    let num_part = s[2..]
                        .split(|c: char| c == 'i' || c == 'u')
                        .next()
                        .unwrap_or("0")
                        .replace('_', "");
                    match i128::from_str_radix(&num_part, 16) {
                        Ok(v) => v,
                        Err(_) => return Ok(Expr::Error(Span::new(start, end))),
                    }
                } else if s.starts_with("0b") || s.starts_with("0B") {
                    let num_part = s[2..]
                        .split(|c: char| c == 'i' || c == 'u')
                        .next()
                        .unwrap_or("0")
                        .replace('_', "");
                    match i128::from_str_radix(&num_part, 2) {
                        Ok(v) => v,
                        Err(_) => return Ok(Expr::Error(Span::new(start, end))),
                    }
                } else {
                    let num_part = s
                        .split(|c: char| c == 'i' || c == 'u')
                        .next()
                        .unwrap_or("0")
                        .replace('_', "");
                    match num_part.parse::<i128>() {
                        Ok(v) => v,
                        Err(_) => return Ok(Expr::Error(Span::new(start, end))),
                    }
                };
                let expr = Expr::Literal(
                    Literal::Int(crate::ast::IntLit::Small(value)),
                    Span::new(start, end),
                );
                if matches!(self.peek(), Ok(Token::Colon)) {
                    self.advance().ok();
                    let ty = self.parse_type()?;
                    let end = self.span().end;
                    Ok(Expr::TypeAnnotated {
                        expr: Self::alloc_shared(self.arena, expr),
                        ty: Self::alloc_shared(self.arena, ty),
                        span: Span::new(start, end),
                    })
                } else {
                    Ok(expr)
                }
            }
            Ok(Token::Forall) | Ok(Token::Exists) => {
                let quantifier = match self.advance().map_err(|_| unreachable!())? {
                    Token::Forall => Quantifier::Forall,
                    Token::Exists => Quantifier::Exists,
                    _ => unreachable!(),
                };
                let binder = match self.advance() {
                    Ok(Token::Ident(name)) => name,
                    Ok(tok) => {
                        return Err(Diagnostic::error(format!(
                            "expected binder name after {:?}, found {:?}",
                            quantifier, tok
                        ))
                        .with_code_str("E004")
                        .with_help(format!(
                            "`{:?}` must be followed by a binder variable and `in <range>`",
                            quantifier
                        ))
                        .with_suggestion("try `forall i in 0..n: arr[i] > 0`")
                        .with_span(self.span()));
                    }
                    Err(()) => {
                        return Err(Diagnostic::error(format!(
                            "unexpected end of file after {:?}",
                            quantifier
                        ))
                        .with_code_str("E002")
                        .with_help("quantified expression is incomplete")
                        .with_span(self.span()));
                    }
                };
                self.expect(Token::In)?;
                let range = self.parse_expr()?;
                self.expect(Token::Colon)?;
                let body = self.parse_expr()?;
                let end = self.span().end;
                Ok(Expr::Quantified {
                    quantifier,
                    binder,
                    range: Self::alloc_shared(self.arena, range),
                    body: Self::alloc_shared(self.arena, body),
                    span: Span::new(start, end),
                })
            }
            Ok(Token::At) => {
                self.advance().ok();
                let name = match self.advance() {
                    Ok(Token::Ident(n)) => n,
                    Ok(tok) => {
                        return Err(Diagnostic::error(format!(
                            "expected built-in name after `@`, found {:?}",
                            tok
                        ))
                        .with_span(self.span()));
                    }
                    Err(()) => {
                        return Err(Diagnostic::error("unexpected end of input after `@`")
                            .with_span(self.span()));
                    }
                };
                if name.eq_str("typeInfo") {
                    self.expect(Token::Bang)?;
                    self.expect(Token::LParen)?;
                    let ty = self.parse_type()?;
                    self.expect(Token::RParen)?;
                    let end = self.span().end;
                    Ok(Expr::TypeInfo(
                        Self::alloc_shared(self.arena, ty),
                        Span::new(start, end),
                    ))
                } else if name.eq_str("compile_error") {
                    // The `!` suffix is optional in `@compile_error` per the spec
                    // (the syntax doc omits it but examples use it).
                    if matches!(self.peek(), Ok(Token::Bang)) {
                        self.advance().ok();
                    }
                    self.expect(Token::LParen)?;
                    let msg = match self.advance() {
                        Ok(Token::StringLiteral(Ok(s))) => s,
                        _ => return Err(Diagnostic::error("@compile_error expects a string literal argument, e.g. `@compile_error(\"message\")`")
                            .with_span(self.span())),
                    };
                    self.expect(Token::RParen)?;
                    let end = self.span().end;
                    Ok(Expr::CompileError(msg, Span::new(start, end)))
                } else {
                    let end = self.span().end;
                    Ok(Expr::Ident(
                        Symbol::intern(&format!("@{}", name.as_str())),
                        Span::new(start, end),
                    ))
                }
            }
            _ => {
                let mut diag = Diagnostic::error("expected expression")
                    .with_code_str("E007")
                    .with_span(self.span());
                if let Ok(Token::Semicolon) = self.peek() {
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

    // -------------------------------------------------------------------------
    // Closure
    // -------------------------------------------------------------------------

    pub(super) fn parse_closure(&mut self, start: usize) -> Result<Expr<'input>, Diagnostic> {
        self.advance().ok();
        let mut params = Vec::new();
        loop {
            if matches!(self.peek(), Ok(Token::Pipe)) {
                self.advance().ok();
                break;
            }
            let name = match self.advance() {
                Ok(Token::Ident(n)) => n,
                Ok(tok) => return Err(Diagnostic::error(format!(
                    "expected parameter name, found {:?}",
                    tok
                ))
                .with_code_str("E004")
                .with_help("closure parameters must have a name — e.g. `|x, y| { ... }`")
                .with_suggestion(
                    "use a valid identifier like `x`, `acc`, or `item` for the closure parameter",
                )
                .with_span(self.span())),
                Err(()) => return Err(Diagnostic::error("unexpected end of file in closure")
                    .with_code_str("E002")
                    .with_help("closure definition is incomplete — expected `| ... | { ... }`")
                    .with_suggestion(
                        "close the closure with `| { body }` or add parameters: `|x, y| { ... }`",
                    )
                    .with_span(self.span())),
            };
            let ty = if matches!(self.peek(), Ok(Token::Colon)) {
                self.advance().ok();
                Some(self.parse_type()?)
            } else {
                None
            };
            params.push(Param {
                name,
                ty,
                default: None,
                span: self.span(),
            });
            if matches!(self.peek(), Ok(Token::Comma)) {
                self.advance().ok();
            } else {
                self.expect(Token::Pipe)?;
                break;
            }
        }
        let return_type = if matches!(self.peek(), Ok(Token::Arrow)) {
            self.advance().ok();
            Some(self.parse_type()?)
        } else {
            None
        };
        let body = if matches!(self.peek(), Ok(Token::LBrace)) {
            self.advance().ok();
            let stmts =
                self.with_restrictions(ParseRestrictions::VALUE_BLOCK, |this| this.parse_block())?;
            self.expect(Token::RBrace)?;
            stmts
        } else {
            let expr = self.parse_expr()?;
            vec![Stmt::Expression(expr)]
        };
        let end = self.span().end;
        Ok(Expr::Closure {
            params,
            return_type,
            captures: Vec::new(),
            body,
            span: Span::new(start, end),
        })
    }

    // -------------------------------------------------------------------------
    // Path / literal / struct / enum / call
    // -------------------------------------------------------------------------

    fn parse_path_or_literal(&mut self, start: usize) -> Result<Expr<'input>, Diagnostic> {
        let mut path = Vec::new();
        let name = match self.advance() {
            Ok(Token::Ident(n)) => n,
            _ => unreachable!(),
        };
        path.push(name);
        while matches!(self.peek(), Ok(Token::ColonColon)) {
            self.advance().ok();
            match self.advance() {
                Ok(tok) if tok.as_ident_symbol().is_some() => {
                    path.push(tok.as_ident_symbol().expect("guarded by is_some() above"))
                }
                _ => return Err(Diagnostic::error("expected identifier after '::'")
                    .with_code_str("E004")
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
            if !matches!(self.peek(), Ok(Token::Bang)) {
                let found = match self.peek() {
                    Ok(tok) => format!("{:?}", tok),
                    Err(()) => "end of file".to_string(),
                };
                return Err(Diagnostic::error(
                    "`layout_of` requires `!` suffix — use `layout_of!(Type)`",
                )
                .with_code_str("E001")
                .with_span(Span::new(start, self.span().end))
                .with_help(format!(
                    "`layout_of` is a comptime intrinsic; add `!` before `(` (found `{}`)",
                    found
                ))
                .with_suggestion("layout_of!(Type)"));
            }
            self.advance().ok();
            self.expect(Token::LParen)?;
            let ty = self.parse_type()?;
            self.expect(Token::RParen)?;
            let end = self.span().end;
            return Ok(Expr::LayoutOf(
                Self::alloc_shared(self.arena, ty),
                Span::new(start, end),
            ));
        }

        match self.peek() {
            Ok(Token::LBrace) if !restrict => self.parse_struct_lit(path, start),
            Ok(Token::LParen) => {
                if path.len() == 2 {
                    let variant = path[1];
                    let enum_path = vec![path[0]];
                    self.parse_enum_lit(enum_path, variant, start)
                } else if path.len() >= 2 {
                    let span = Span::new(start, self.span().end);
                    let callee = Expr::Path(smallvec::SmallVec::from(path), span);
                    self.parse_call(callee, start)
                } else {
                    self.parse_call(
                        Expr::Ident(
                            path.into_iter()
                                .next()
                                .expect("expected at least one path segment"),
                            Span::new(start, self.span().start),
                        ),
                        start,
                    )
                }
            }
            _ => {
                if path.len() >= 2 {
                    let variant = path.pop().expect("Enum pattern must have a variant");
                    let end = self.span().end;
                    Ok(Expr::EnumLit {
                        path,
                        variant,
                        payload: None,
                        span: Span::new(start, end),
                    })
                } else {
                    Ok(Expr::Ident(
                        path.into_iter()
                            .next()
                            .expect("expected at least one path segment"),
                        Span::new(start, self.span().end),
                    ))
                }
            }
        }
    }

    fn parse_struct_lit(
        &mut self,
        path: Vec<Symbol>,
        start: usize,
    ) -> Result<Expr<'input>, Diagnostic> {
        self.expect(Token::LBrace)?;
        let mut fields = Vec::new();
        loop {
            if matches!(self.peek(), Ok(Token::RBrace)) {
                self.advance().ok();
                break;
            }
            let field_name = match self.advance() {
                Ok(Token::Ident(n)) => n,
                Ok(tok) => return Err(Diagnostic::error(format!(
                    "expected field name, found {:?}",
                    tok
                ))
                .with_code_str("E004")
                .with_help("struct literal fields must have a name — e.g. `Point { x = 1, y = 2 }`")
                .with_suggestion("add a field name followed by `= <value>`, e.g. `field = value`")
                .with_span(self.span())),
                Err(()) => {
                    return Err(
                        Diagnostic::error("unexpected end of file in struct literal")
                            .with_code_str("E002")
                            .with_help(
                                "struct literal is incomplete — expected a field name or `}`",
                            )
                            .with_suggestion("close the struct with `}` or add more fields")
                            .with_span(self.span()),
                    );
                }
            };
            self.expect(Token::Assign)?;
            let value = self.parse_expr()?;
            fields.push((field_name, value));
            if matches!(self.peek(), Ok(Token::Comma)) {
                self.advance().ok();
            } else {
                self.expect(Token::RBrace)?;
                break;
            }
        }
        let end = self.span().end;
        Ok(Expr::StructLit {
            path,
            fields,
            span: Span::new(start, end),
        })
    }

    fn parse_enum_lit(
        &mut self,
        path: Vec<Symbol>,
        variant: Symbol,
        start: usize,
    ) -> Result<Expr<'input>, Diagnostic> {
        self.expect(Token::LParen)?;
        if matches!(self.peek(), Ok(Token::RParen)) {
            self.advance().ok();
            let end = self.span().end;
            return Ok(Expr::EnumLit {
                path,
                variant,
                payload: None,
                span: Span::new(start, end),
            });
        }
        let payload = self.parse_expr()?;
        if matches!(self.peek(), Ok(Token::Comma)) {
            let mut args = vec![payload];
            while matches!(self.peek(), Ok(Token::Comma)) {
                self.advance().ok();
                args.push(self.parse_expr()?);
            }
            self.expect(Token::RParen)?;
            let end = self.span().end;
            let callee = Expr::Path(smallvec::SmallVec::from(path), Span::new(start, end));
            return Ok(Expr::Call {
                callee: Self::alloc_shared(self.arena, callee),
                args,
                comptime: false,
                span: Span::new(start, end),
            });
        }
        self.expect(Token::RParen)?;
        let end = self.span().end;
        Ok(Expr::EnumLit {
            path,
            variant,
            payload: Some(Self::alloc_shared(self.arena, payload)),
            span: Span::new(start, end),
        })
    }

    fn parse_call(
        &mut self,
        callee: Expr<'input>,
        start: usize,
    ) -> Result<Expr<'input>, Diagnostic> {
        self.expect(Token::LParen)?;
        let mut args = Vec::new();
        if !matches!(self.peek(), Ok(Token::RParen)) {
            loop {
                args.push(self.parse_expr()?);
                if matches!(self.peek(), Ok(Token::Comma)) {
                    self.advance().ok();
                } else {
                    break;
                }
            }
        }
        self.expect(Token::RParen)?;
        let end = self.span().end;
        Ok(Expr::Call {
            callee: Self::alloc_shared(self.arena, callee),
            args,
            comptime: false,
            span: Span::new(start, end),
        })
    }

    // -------------------------------------------------------------------------
    // Literal parsing
    // -------------------------------------------------------------------------

    pub(super) fn parse_literal(&mut self) -> Result<Expr<'input>, Diagnostic> {
        let start = self.span().start;
        let token = self.advance().map_err(|_| {
            Diagnostic::error("unexpected token")
                .with_code_str("E003")
                .with_help("expected a literal value (number, string, char, bool, or byte string)")
                .with_span(Span::new(0, 0))
        })?;
        let end = self.span().end;
        let span = Span::new(start, end);
        match token {
            Token::IntLiteral(res) => match res {
                Ok(v) => Ok(Expr::Literal(
                    Literal::Int(crate::ast::IntLit::Small(v)),
                    span,
                )),
                Err(e) => {
                    self.diagnostics.push(Diagnostic::error(e).with_span(span));
                    Ok(Expr::Error(span))
                }
            },
            Token::HexLiteral(res) => match res {
                Ok(v) => Ok(Expr::Literal(
                    Literal::Int(crate::ast::IntLit::Small(v)),
                    span,
                )),
                Err(e) => {
                    self.diagnostics.push(Diagnostic::error(e).with_span(span));
                    Ok(Expr::Error(span))
                }
            },
            Token::BinLiteral(res) => match res {
                Ok(v) => Ok(Expr::Literal(
                    Literal::Int(crate::ast::IntLit::Small(v)),
                    span,
                )),
                Err(e) => {
                    self.diagnostics.push(Diagnostic::error(e).with_span(span));
                    Ok(Expr::Error(span))
                }
            },
            Token::FloatLiteral(res) => match res {
                Ok(v) => Ok(Expr::Literal(Literal::Float(v), span)),
                // The committee ruling (float default `trap` — aligned
                // with integers): a compile-time float-literal anomaly
                // (overflow/NaN/inf) is a COMPILE ERROR — the error must
                // not be swallowed into Expr::Error.
                Err(e) => {
                    return Err(Diagnostic::error(e).with_span(span));
                }
            },
            Token::CharLiteral(res) => match res {
                Ok(v) => Ok(Expr::Literal(Literal::Char(v), span)),
                Err(e) => {
                    self.diagnostics.push(Diagnostic::error(e).with_span(span));
                    Ok(Expr::Error(span))
                }
            },
            Token::StringLiteral(res) => match res {
                Ok(s) => Ok(Expr::Literal(Literal::String(s), span)),
                Err(e) => {
                    self.diagnostics.push(Diagnostic::error(e).with_span(span));
                    Ok(Expr::Error(span))
                }
            },
            Token::ByteStringLiteral(res) => match res {
                Ok(b) => Ok(Expr::Literal(Literal::ByteString(b), span)),
                Err(e) => {
                    self.diagnostics.push(Diagnostic::error(e).with_span(span));
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

    fn parse_infix(&mut self, lhs: Expr<'input>, bp: u8) -> Result<Expr<'input>, Diagnostic> {
        let start = self.span().start;
        match self.peek() {
            Ok(Token::Bang) => {
                if matches!(lhs, Expr::Ident(..) | Expr::Path(..))
                    && matches!(self.peek_next(), Some(Token::LParen))
                {
                    self.advance().ok();
                    self.advance().ok();
                    let mut args = Vec::new();
                    if !matches!(self.peek(), Ok(Token::RParen)) {
                        loop {
                            args.push(self.parse_expr()?);
                            if matches!(self.peek(), Ok(Token::Comma)) {
                                self.advance().ok();
                            } else {
                                break;
                            }
                        }
                    }
                    self.expect(Token::RParen)?;
                    Ok(Expr::Call {
                        callee: Self::alloc_shared(self.arena, lhs),
                        args,
                        comptime: true,
                        span: Span::new(start, self.span().end),
                    })
                } else {
                    Err(Diagnostic::error("unexpected !")
                        .with_code_str("E007")
                        .with_help("`!` as a postfix operator is only used for comptime calls — e.g. `func!()`")
                        .with_suggestion("use `func!()` for comptime calls, or remove the `!`")
                        .with_span(self.span()))
                }
            }
            Ok(Token::Question) => {
                self.advance().ok();
                let end = self.span().end;
                Ok(Expr::Try {
                    expr: Self::alloc_shared(self.arena, lhs),
                    span: Span::new(start, end),
                })
            }
            Ok(Token::As) => {
                self.advance().ok();
                let safe = !matches!(self.peek(), Ok(Token::Bang));
                if !safe {
                    self.advance().ok();
                }
                let ty = self.parse_type()?;
                let rounding = match self.peek() {
                    Ok(Token::Round) => {
                        self.advance().ok();
                        Some(Rounding::Round)
                    }
                    Ok(Token::Trunc) => {
                        self.advance().ok();
                        Some(Rounding::Trunc)
                    }
                    Ok(Token::Ceil) => {
                        self.advance().ok();
                        Some(Rounding::Ceil)
                    }
                    Ok(Token::Floor) => {
                        self.advance().ok();
                        Some(Rounding::Floor)
                    }
                    _ => None,
                };
                let end = self.span().end;
                Ok(Expr::Cast {
                    expr: Self::alloc_shared(self.arena, lhs),
                    ty: Self::alloc_shared(self.arena, ty),
                    safe,
                    rounding,
                    span: Span::new(start, end),
                })
            }
            Ok(Token::Plus) => {
                self.advance().ok();
                self.binary(lhs, BinOp::Add, bp + 1, start)
            }
            Ok(Token::Minus) => {
                self.advance().ok();
                self.binary(lhs, BinOp::Sub, bp + 1, start)
            }
            Ok(Token::Star) => {
                self.advance().ok();
                self.binary(lhs, BinOp::Mul, bp + 1, start)
            }
            Ok(Token::Slash) => {
                self.advance().ok();
                self.binary(lhs, BinOp::Div, bp + 1, start)
            }
            Ok(Token::Percent) => {
                self.advance().ok();
                self.binary(lhs, BinOp::Rem, bp + 1, start)
            }
            Ok(Token::PlusWrap) => {
                self.advance().ok();
                self.binary(lhs, BinOp::AddWrap, bp + 1, start)
            }
            Ok(Token::MinusWrap) => {
                self.advance().ok();
                self.binary(lhs, BinOp::SubWrap, bp + 1, start)
            }
            Ok(Token::StarWrap) => {
                self.advance().ok();
                self.binary(lhs, BinOp::MulWrap, bp + 1, start)
            }
            Ok(Token::PlusSaturate) => {
                self.advance().ok();
                self.binary(lhs, BinOp::AddSaturate, bp + 1, start)
            }
            Ok(Token::MinusSaturate) => {
                self.advance().ok();
                self.binary(lhs, BinOp::SubSaturate, bp + 1, start)
            }
            Ok(Token::StarSaturate) => {
                self.advance().ok();
                self.binary(lhs, BinOp::MulSaturate, bp + 1, start)
            }
            Ok(Token::PlusTrap) => {
                self.advance().ok();
                self.binary(lhs, BinOp::AddTrap, bp + 1, start)
            }
            Ok(Token::MinusTrap) => {
                self.advance().ok();
                self.binary(lhs, BinOp::SubTrap, bp + 1, start)
            }
            Ok(Token::StarTrap) => {
                self.advance().ok();
                self.binary(lhs, BinOp::MulTrap, bp + 1, start)
            }
            Ok(Token::Ampersand) => {
                self.advance().ok();
                self.binary(lhs, BinOp::BitAnd, bp + 1, start)
            }
            Ok(Token::Pipe) => {
                self.advance().ok();
                self.binary(lhs, BinOp::BitOr, bp + 1, start)
            }
            Ok(Token::Caret) => {
                self.advance().ok();
                self.binary(lhs, BinOp::BitXor, bp + 1, start)
            }
            Ok(Token::Shl) => {
                self.advance().ok();
                self.binary(lhs, BinOp::Shl, bp + 1, start)
            }
            Ok(Token::Shr) => {
                self.advance().ok();
                self.binary(lhs, BinOp::Shr, bp + 1, start)
            }
            Ok(Token::EqEq) => {
                self.advance().ok();
                self.binary(lhs, BinOp::Eq, bp + 1, start)
            }
            Ok(Token::Neq) => {
                self.advance().ok();
                self.binary(lhs, BinOp::Neq, bp + 1, start)
            }
            Ok(Token::Lt) => {
                self.advance().ok();
                self.binary(lhs, BinOp::Lt, bp + 1, start)
            }
            Ok(Token::Gt) => {
                self.advance().ok();
                self.binary(lhs, BinOp::Gt, bp + 1, start)
            }
            Ok(Token::Le) => {
                self.advance().ok();
                self.binary(lhs, BinOp::Le, bp + 1, start)
            }
            Ok(Token::Ge) => {
                self.advance().ok();
                self.binary(lhs, BinOp::Ge, bp + 1, start)
            }
            Ok(Token::And) => {
                self.advance().ok();
                self.binary(lhs, BinOp::And, bp + 1, start)
            }
            Ok(Token::Or) => {
                self.advance().ok();
                self.binary(lhs, BinOp::Or, bp + 1, start)
            }
            Ok(Token::DotDot) => {
                self.advance().ok();
                let end = if !matches!(
                    self.peek(),
                    Ok(Token::Semicolon) | Ok(Token::RBrace) | Ok(Token::Comma) | Ok(Token::RParen)
                ) {
                    Some(Self::alloc_shared(self.arena, self.parse_expr_bp(0)?))
                } else {
                    None
                };
                Ok(Expr::Range {
                    start: Some(Self::alloc_shared(self.arena, lhs)),
                    end,
                    inclusive: false,
                    span: Span::new(start, self.span().end),
                })
            }
            Ok(Token::DotDotEq) => {
                self.advance().ok();
                let end = self.parse_expr_bp(0)?;
                Ok(Expr::Range {
                    start: Some(Self::alloc_shared(self.arena, lhs)),
                    end: Some(Self::alloc_shared(self.arena, end)),
                    inclusive: true,
                    span: Span::new(start, self.span().end),
                })
            }
            Ok(Token::LParen) => {
                self.advance().ok();
                let mut args = Vec::new();
                if !matches!(self.peek(), Ok(Token::RParen)) {
                    loop {
                        args.push(self.parse_expr()?);
                        if matches!(self.peek(), Ok(Token::Comma)) {
                            self.advance().ok();
                        } else {
                            break;
                        }
                    }
                }
                self.expect(Token::RParen)?;
                Ok(Expr::Call {
                    callee: Self::alloc_shared(self.arena, lhs),
                    args,
                    comptime: false,
                    span: Span::new(start, self.span().end),
                })
            }
            Ok(Token::LBracket) => {
                self.advance().ok();
                let index = self.parse_expr()?;
                self.expect(Token::RBracket)?;
                Ok(Expr::Index {
                    base: Self::alloc_shared(self.arena, lhs),
                    index: Self::alloc_shared(self.arena, index),
                    span: Span::new(start, self.span().end),
                })
            }
            Ok(Token::Dot) => {
                self.advance().ok();
                if let Ok(Token::Ident(field)) = self.advance() {
                    if field.eq_str("freeze") && matches!(self.peek(), Ok(Token::Bang)) {
                        // `.freeze!()` — explicit read-only freeze, equivalent
                        // to `&ro expr` (SYNTAX.md: ".freeze!() ... behaves
                        // identically and is preferred in method chains").
                        self.advance().ok();
                        self.expect(Token::LParen)?;
                        self.expect(Token::RParen)?;
                        Ok(Expr::UnaryOp {
                            op: crate::ast::UnaryOp::Ro,
                            expr: Self::alloc_shared(self.arena, lhs),
                            span: Span::new(start, self.span().end),
                        })
                    } else {
                        Ok(Expr::FieldAccess {
                            base: Self::alloc_shared(self.arena, lhs),
                            field,
                            span: Span::new(start, self.span().end),
                        })
                    }
                } else {
                    Err(Diagnostic::error("expected field name after '.'")
                        .with_code_str("E004")
                        .with_help("`.` must be followed by a field name — e.g. `object.field`")
                        .with_suggestion("add a field name after `.`, or remove the `.`")
                        .with_span(self.span()))
                }
            }
            Ok(Token::Apostrophe) => {
                self.advance().ok();
                if let Ok(Token::Ident(attr)) = self.advance() {
                    Ok(Expr::AttrAccess {
                        base: Self::alloc_shared(self.arena, lhs),
                        attr,
                        span: Span::new(start, self.span().end),
                    })
                } else {
                    Err(Diagnostic::error("expected attribute name after '''")
                        .with_code_str("E004")
                        .with_help("`'` must be followed by an attribute name — e.g. `object'attr`")
                        .with_suggestion("add an attribute name after `'`, or remove the `'`")
                        .with_span(self.span()))
                }
            }
            Ok(Token::Catch) => {
                self.advance().ok();
                let mut branches = Vec::new();
                self.expect(Token::LBrace)?;
                loop {
                    if matches!(self.peek(), Ok(Token::RBrace)) {
                        self.advance().ok();
                        break;
                    }
                    let branch_start = self.span().start;
                    self.expect(Token::Pipe)?;
                    let pattern = self.parse_pattern()?;
                    let bind = if matches!(self.peek(), Ok(Token::As)) {
                        self.advance().ok();
                        match self.advance() {
                            Ok(Token::Ident(name)) => Some(name),
                            _ => return Err(Diagnostic::error("expected binding name after 'as'")
                                .with_code_str("E004")
                                .with_help("`as` in a catch pattern must be followed by a binding name — e.g. `|NetworkError as e|`")
                                .with_suggestion("add a capture variable name after `as`, like `|Pattern as var_name|`")
                                .with_span(self.span())),
                        }
                    } else {
                        None
                    };
                    self.expect(Token::Pipe)?;
                    let body = if matches!(self.peek(), Ok(Token::FatArrow)) {
                        self.advance().ok();
                        let expr = self.parse_expr()?;
                        vec![Stmt::Expression(expr)]
                    } else {
                        self.expect(Token::LBrace)?;
                        let block = self
                            .with_restrictions(ParseRestrictions::VALUE_BLOCK, |this| {
                                this.parse_block()
                            })?;
                        self.expect(Token::RBrace)?;
                        block
                    };
                    branches.push(CatchBranch {
                        pattern,
                        bind,
                        body,
                        span: Span::new(branch_start, self.span().end),
                    });
                }
                Ok(Expr::Catch {
                    expr: Self::alloc_shared(self.arena, lhs),
                    branches,
                    span: Span::new(start, self.span().end),
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
        start: usize,
    ) -> Result<Expr<'input>, Diagnostic> {
        let rhs = self.parse_expr_bp(bp)?;
        Ok(Expr::BinaryOp {
            left: Self::alloc_shared(self.arena, lhs),
            op,
            right: Self::alloc_shared(self.arena, rhs),
            span: Span::new(start, self.span().end),
        })
    }

    // -------------------------------------------------------------------------
    // `if` expression
    // -------------------------------------------------------------------------

    fn parse_if_expr(&mut self) -> Result<Expr<'input>, Diagnostic> {
        let start = self.span().start;
        self.advance().ok();
        if matches!(self.peek(), Ok(Token::Let)) {
            self.advance().ok();
            let pattern = self.parse_or_pattern()?;
            self.expect(Token::Assign)?;
            let scrutinee = self
                .with_restrictions(ParseRestrictions::NO_STRUCT_LITERAL, |this| {
                    this.parse_expr()
                })?;
            self.expect(Token::LBrace)?;
            let then_branch =
                self.with_restrictions(ParseRestrictions::VALUE_BLOCK, |this| this.parse_block())?;
            self.expect(Token::RBrace)?;
            let else_branch = if matches!(self.peek(), Ok(Token::Else)) {
                self.advance().ok();
                if matches!(self.peek(), Ok(Token::If)) {
                    Some(vec![Stmt::Expression(self.parse_if_expr()?)])
                } else {
                    self.expect(Token::LBrace)?;
                    let block = self.with_restrictions(ParseRestrictions::VALUE_BLOCK, |this| {
                        this.parse_block()
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
                scrutinee: Self::alloc_shared(self.arena, scrutinee),
                then_branch,
                else_branch,
                is_expression: true,
                span: Span::new(start, end),
            });
        }
        let cond = self.with_restrictions(ParseRestrictions::NO_STRUCT_LITERAL, |this| {
            this.parse_expr()
        })?;
        self.expect(Token::LBrace)?;
        let then_branch =
            self.with_restrictions(ParseRestrictions::VALUE_BLOCK, |this| this.parse_block())?;
        self.expect(Token::RBrace)?;
        let else_branch = if matches!(self.peek(), Ok(Token::Else)) {
            self.advance().ok();
            if matches!(self.peek(), Ok(Token::If)) {
                Some(vec![Stmt::Expression(self.parse_if_expr()?)])
            } else {
                self.expect(Token::LBrace)?;
                let block = self
                    .with_restrictions(ParseRestrictions::VALUE_BLOCK, |this| this.parse_block())?;
                self.expect(Token::RBrace)?;
                Some(block)
            }
        } else {
            None
        };
        let end = self.span().end;
        Ok(Expr::If {
            cond: Self::alloc_shared(self.arena, cond),
            then_branch,
            else_branch,
            is_expression: true,
            span: Span::new(start, end),
        })
    }

    // -------------------------------------------------------------------------
    // Match expression
    // -------------------------------------------------------------------------

    pub(super) fn parse_match_expr(&mut self) -> Result<Expr<'input>, Diagnostic> {
        let start = self.span().start;
        self.advance().ok();
        let scrutinee = self.with_restrictions(ParseRestrictions::NO_STRUCT_LITERAL, |this| {
            this.parse_expr()
        })?;
        self.expect(Token::LBrace)?;
        let mut arms = Vec::new();
        loop {
            if matches!(self.peek(), Ok(Token::RBrace)) {
                self.advance().ok();
                break;
            }
            let arm_start = self.span().start;
            let pattern = self.parse_or_pattern()?;
            let guard = if matches!(self.peek(), Ok(Token::If)) {
                self.advance().ok();
                Some(Self::alloc_shared(self.arena, self.parse_expr()?))
            } else {
                None
            };
            self.expect(Token::FatArrow)?;
            let body = self.with_restrictions(
                ParseRestrictions::VALUE_BLOCK | ParseRestrictions::NO_STRUCT_LITERAL,
                |this| this.parse_expr(),
            )?;
            arms.push(MatchArm {
                pattern,
                guard,
                body,
                span: Span::new(arm_start, self.span().end),
            });
            if matches!(self.peek(), Ok(Token::Comma)) {
                self.advance().ok();
            }
            if matches!(self.peek(), Ok(Token::RBrace)) {
                self.advance().ok();
                break;
            }
        }
        let end = self.span().end;
        Ok(Expr::Match {
            scrutinee: Self::alloc_shared(self.arena, scrutinee),
            arms,
            span: Span::new(start, end),
        })
    }
}
