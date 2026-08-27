//! Parsing of type expressions: `&T`, `*T`, `[T]`, `(A,B)`, `fn types`, projections, etc.

use super::*;
use crate::ast::*;
use crate::diagnostics::Diagnostic;
use crate::lexer::Token;
use crate::symbol::Symbol;
use smallvec::smallvec;

impl<'input> Parser<'input> {
    // -------------------------------------------------------------------------
    // Type entry with recursion guard
    // -------------------------------------------------------------------------

    pub fn parse_type(&mut self) -> Result<Type<'input>, Diagnostic> {
        self.recursion_depth += 1;
        if self.recursion_depth > self.max_recursion_depth {
            self.recursion_depth -= 1;
            return Err(Diagnostic::error(format!("maximum recursion depth {} exceeded", self.max_recursion_depth))
                .with_code_str("E006")
                .with_help("the parser reached its recursion limit — the type/structure may be deeply nested or self-referential")
                .with_suggestion("try breaking up deeply nested structures, or use less complex type expressions")
                .with_span(self.span()));
        }
        let result = self.parse_type_inner();
        self.recursion_depth -= 1;
        result
    }

    // -------------------------------------------------------------------------
    // Inner type parser (Pratt‑style)
    // -------------------------------------------------------------------------

    fn parse_type_inner(&mut self) -> Result<Type<'input>, Diagnostic> {
        let start = self.span().start;
        match self.peek() {
            Ok(Token::For) => {
                // Higher-ranked type: `for<'a> T` (SYNTAX.md
                // §Higher-Ranked Trait Bounds — "for<'a> introduces one or
                // more lifetime parameters scoped over the subsequent
                // trait bound").  The lifetime is universally quantified
                // over the body; the checker skolemizes it at the call
                // site.
                self.advance().ok();
                self.expect(Token::Lt)?;
                self.expect(Token::Apostrophe)?;
                let lifetime = match self.advance() {
                    Ok(Token::Ident(name)) => name,
                    _ => return Err(Diagnostic::error("expected a lifetime name after `for<'`")
                        .with_code_str("E004")
                        .with_span(self.span())),
                };
                self.expect_gt()?;
                let body = Self::alloc_shared(self.arena, self.parse_type()?);
                let end = self.span().end;
                Ok(Type::Forall { lifetime, body, span: Span::new(start, end) })
            }
            Ok(Token::Lt) => {
                // Qualified path / projection: `<ImplType as TraitPath>::AssocName`
                self.advance().ok();
                let impl_type = Self::alloc_shared(self.arena, self.parse_type()?);
                self.expect(Token::As)?;
                let trait_path = Self::alloc_shared(self.arena, self.parse_type()?);
                self.expect_gt()?;
                self.expect(Token::ColonColon)?;
                let assoc_name = match self.advance() {
                    Ok(Token::Ident(name)) => name,
                    _ => return Err(Diagnostic::error("expected associated type name after `::`")
                        .with_code_str("E004")
                        .with_help("qualified paths use `<Type as Trait>::AssocType` syntax")
                        .with_suggestion("add the associated type name after `::`, e.g. `<T as Display>::Output`")
                        .with_span(self.span())),
                };
                let end = self.span().end;
                Ok(Type::Projection { impl_type, trait_path, assoc_name, span: Span::new(start, end) })
            }
            Ok(Token::Shl) => {
                // Nested projection starts with `<<`: `<<A as Trait1>::X as Trait2>::Y`
                // The lexer merged `<<` into Shl; push one `Lt` back and treat as `<`.
                self.advance().ok();
                self.pending.push(Token::Lt);
                let impl_type = Self::alloc_shared(self.arena, self.parse_type()?);
                self.expect(Token::As)?;
                let trait_path = Self::alloc_shared(self.arena, self.parse_type()?);
                self.expect_gt()?;
                self.expect(Token::ColonColon)?;
                let assoc_name = match self.advance() {
                    Ok(Token::Ident(name)) => name,
                    _ => return Err(Diagnostic::error("expected associated type name after `::`")
                        .with_code_str("E004")
                        .with_help("qualified paths use `<Type as Trait>::AssocType` syntax")
                        .with_suggestion("add the associated type name after `::`, e.g. `<T as Display>::Output`")
                        .with_span(self.span())),
                };
                let end = self.span().end;
                Ok(Type::Projection { impl_type, trait_path, assoc_name, span: Span::new(start, end) })
            }
            Ok(Token::Ampersand) => {
                self.advance().ok();
                let lifetime = if matches!(self.peek(), Ok(Token::Apostrophe)) {
                    self.advance().ok();
                    match self.advance() {
                        Ok(Token::Ident(name)) => Some(name),
                        _ => return Err(Diagnostic::error("expected lifetime name after `'`")
                            .with_code_str("E004")
                            .with_help("lifetimes use `'name` syntax — e.g. `&'a T`")
                            .with_suggestion("add a lifetime name after `'`, e.g. `&'a mut T`")
                            .with_span(self.span())),
                    }
                } else { None };
                let mutable = matches!(self.peek(), Ok(Token::Mut));
                if mutable { self.advance().ok(); }
                let ty = self.parse_type()?;
                let end = self.span().end;
                Ok(Type::Reference { inner: Self::alloc_shared(self.arena, ty), mutable, lifetime, span: Span::new(start, end) })
            }
            Ok(Token::Star) => {
                self.advance().ok();
                let ty = self.parse_type()?;
                let end = self.span().end;
                Ok(Type::Pointer(Self::alloc_shared(self.arena, ty), Span::new(start, end)))
            }
            Ok(Token::LBracket) => {
                self.advance().ok();
                let ty = self.parse_type()?;
                if matches!(self.peek(), Ok(Token::Semicolon)) {
                    self.advance().ok();
                    let size = self.parse_expr()?;
                    self.expect(Token::RBracket)?;
                    let end = self.span().end;
                    Ok(Type::Array(Self::alloc_shared(self.arena, ty), Self::alloc_shared(self.arena, size), Span::new(start, end)))
                } else {
                    self.expect(Token::RBracket)?;
                    let end = self.span().end;
                    Ok(Type::Slice(Self::alloc_shared(self.arena, ty), Span::new(start, end)))
                }
            }
            Ok(Token::Dyn) => {
                self.advance().ok();
                let mut traits = Vec::new();
                loop {
                    traits.push(self.parse_type()?);
                    if !matches!(self.peek(), Ok(Token::Plus)) { break; }
                    self.advance().ok();
                }
                let end = self.span().end;
                Ok(Type::DynTrait(traits, Span::new(start, end)))
            }
            Ok(Token::Exists) => {
                self.advance().ok();
                let name = match self.advance() {
                    Ok(Token::Ident(n)) => n,
                    _ => return Err(Diagnostic::error("expected identifier after exists")
                        .with_code_str("E004")
                        .with_help("`exists` must be followed by a bound variable name — e.g. `exists n: T invariant ...`")
                        .with_suggestion("add a bound variable name, e.g. `exists n: Int<32> invariant n > 0`")
                        .with_span(self.span())),
                };
                self.expect(Token::Colon)?;
                let base = self.parse_type()?;
                self.expect(Token::Invariant)?;
                let invariant = Self::alloc_shared(self.arena, self.parse_expr()?);
                let end = self.span().end;
                Ok(Type::Exists { name, base: Self::alloc_shared(self.arena, base), invariant, span: Span::new(start, end) })
            }
            Ok(Token::IntLiteral(_)) | Ok(Token::HexLiteral(_)) | Ok(Token::BinLiteral(_)) => {
                let expr = self.parse_literal()?;
                let end = self.span().end;
                Ok(Type::Literal(Self::alloc_shared(self.arena, expr), Span::new(start, end)))
            }
            Ok(Token::Type) => {
                self.advance().ok();
                let end = self.span().end;
                Ok(Type::Path(smallvec![Symbol::intern("type")], Span::new(start, end)))
            }
            _ => match self.advance() {
                Ok(Token::Ident(name)) => {
                    // Special case: `Regex<"pattern">` — compile-time regex type.
                    // (SYNTAX.md §Compile-Time Regular Expressions)
                    if name.eq_str("Regex") && matches!(self.peek(), Ok(Token::Lt)) {
                        self.advance().ok();
                        let pattern = match self.advance() {
                            Ok(Token::StringLiteral(Ok(s))) => s,
                            Ok(Token::StringLiteral(Err(e))) => return Err(Diagnostic::error(format!("invalid string literal in regex pattern: {}", e))
                                .with_code_str("E004")
                                .with_span(self.span())),
                            Ok(tok) => return Err(Diagnostic::error(format!("expected string literal as regex pattern, found {:?}", tok))
                                .with_code_str("E004")
                                .with_help("`Regex<\"...\">` requires a string literal pattern — e.g. `Regex<\"[0-9]+\">`")
                                .with_suggestion("use a string literal like `\"[0-9a-fA-F]+\"` as the regex pattern")
                                .with_span(self.span())),
                            Err(()) => return Err(Diagnostic::error("unexpected end of file in regex pattern")
                                .with_code_str("E002")
                                .with_help("expected a string literal pattern after `<` in `Regex<\"...\">`")
                                .with_suggestion("add a string literal pattern, e.g. `Regex<\"[0-9]+\">`")
                                .with_span(self.span())),
                        };
                        if let Err(e) = regex_syntax::parse(&pattern) {
                            return Err(Diagnostic::error(format!("invalid regex pattern: {}", e))
                                .with_code_str("E004")
                                .with_help("`Regex<\"...\">` requires a valid regular expression pattern")
                                .with_suggestion("check the regex syntax — see https://docs.rs/regex/latest/regex/#syntax")
                                .with_span(self.span()));
                        }
                        self.expect(Token::Gt)?;
                        let end = self.span().end;
                        return Ok(Type::Regex(pattern, Span::new(start, end)));
                    }

                    // Named function type: `Fn(T1, T2) -> R` (SYNTAX.md
                    // §Higher-Ranked Trait Bounds — `where F: for<'a> Fn(&'a
                    // T) -> &'a T`).  Parses like the `(A, B) -> C`
                    // function type below.
                    if name.eq_str("Fn") && matches!(self.peek(), Ok(Token::LParen)) {
                        self.advance().ok();
                        let mut params = Vec::new();
                        if !matches!(self.peek(), Ok(Token::RParen)) {
                            loop {
                                params.push(self.parse_type()?);
                                match self.peek() {
                                    Ok(Token::Comma) => { self.advance().ok(); }
                                    Ok(Token::RParen) => { self.advance().ok(); break; }
                                    _ => return Err(Diagnostic::error("expected ',' or ')' in Fn parameter list")
                                        .with_code_str("E004")
                                        .with_span(self.span())),
                                }
                            }
                        } else {
                            self.advance().ok();
                        }
                        self.expect(Token::Arrow)?;
                        let ret = Self::alloc_shared(self.arena, self.parse_type()?);
                        let end = self.span().end;
                        return Ok(Type::Function { params, ret, span: Span::new(start, end) });
                    }

                    let mut path = vec![name];
                    while matches!(self.peek(), Ok(Token::ColonColon)) {
                        self.advance().ok();
                        if let Ok(Token::Ident(part)) = self.advance() {
                            path.push(part);
                        } else {
                            return Err(Diagnostic::error("expected identifier after '::'")
                                .with_code_str("E004")
                                .with_help("`::` must be followed by an identifier — e.g. `std::collections::HashMap`")
                                .with_suggestion("add an identifier after `::`, e.g. `MyModule::MyType`")
                                .with_span(self.span()));
                        }
                    }
                    if matches!(self.peek(), Ok(Token::Lt)) {
                        self.advance().ok();
                        let mut args = Vec::new();
                        loop {
                            let arg = if matches!(self.peek(), Ok(Token::Ident(_))) && matches!(self.peek_next(), Some(Token::Assign)) {
                                let name = match self.advance() {
                                    Ok(Token::Ident(n)) => n,
                                    _ => unreachable!(),
                                };
                                self.advance().ok(); // consume =
                                let value = if self.check_const_arg() {
                                    let expr = self.with_restrictions(ParseRestrictions::NO_COMPARISON, |this| this.parse_expr())?;
                                    let span = expr.span();
                                    Type::Expr(Self::alloc_shared(self.arena, expr), span)
                                } else {
                                    self.parse_type()?
                                };
                                GenericArg::Named(name, value)
                            } else if self.check_const_arg() {
                                let anon = self.parse_const_arg()?;
                                GenericArg::Const(anon)
                            } else {
                                let cp = self.checkpoint();
                                match self.parse_type() {
                                    Ok(ty) => {
                                        let next_is_shr = matches!(self.peek(), Ok(Token::Shr)) && matches!(self.peek_next(),
                                            Some(Token::IntLiteral(_)) | Some(Token::FloatLiteral(_)) | Some(Token::True) | Some(Token::False)
                                            | Some(Token::CharLiteral(_)) | Some(Token::StringLiteral(_)) | Some(Token::ByteStringLiteral(_))
                                            | Some(Token::Ident(_)) | Some(Token::LParen) | Some(Token::LBracket)
                                            | Some(Token::Minus) | Some(Token::Plus) | Some(Token::Bang) | Some(Token::Tilde));
                                        if next_is_shr || matches!(self.peek(),
                                            Ok(Token::Plus) | Ok(Token::Minus) | Ok(Token::Star) | Ok(Token::Slash) | Ok(Token::Percent)
                                            | Ok(Token::Shl) | Ok(Token::Ampersand) | Ok(Token::Pipe) | Ok(Token::Caret)
                                            | Ok(Token::LParen) | Ok(Token::LBracket) | Ok(Token::Dot) | Ok(Token::Apostrophe) | Ok(Token::Bang))
                                        {
                                            self.restore(&cp);
                                            let expr = self.with_restrictions(ParseRestrictions::NO_COMPARISON, |this| this.parse_expr())?;
                                            if !Self::is_simple_const_expr(&expr) {
                                                let span = expr.span();
                                                return Err(Diagnostic::error("complex const generic argument must be wrapped in `{ }`")
                                                    .with_code_str("E004")
                                                    .with_help("wrap the expression in braces — e.g. `<{ N + 1 }>` instead of `<N + 1>`")
                                                    .with_suggestion("add `{` before and `}` after the expression")
                                                    .with_span(span));
                                            }
                                            let span = expr.span();
                                            GenericArg::Const(AnonConst { value: Self::alloc_shared(self.arena, expr), span })
                                        } else {
                                            GenericArg::Positional(ty)
                                        }
                                    }
                                    Err(_) => {
                                        self.restore(&cp);
                                        let expr = self.with_restrictions(ParseRestrictions::NO_COMPARISON, |this| this.parse_expr())?;
                                        if !Self::is_simple_const_expr(&expr) {
                                            let span = expr.span();
                                            return Err(Diagnostic::error("complex const generic argument must be wrapped in `{ }`")
                                                .with_code_str("E004")
                                                .with_help("wrap the expression in braces — e.g. `<{ N + 1 }>` instead of `<N + 1>`")
                                                .with_suggestion("add `{` before and `}` after the expression")
                                                .with_span(span));
                                        }
                                        let span = expr.span();
                                        GenericArg::Const(AnonConst { value: Self::alloc_shared(self.arena, expr), span })
                                    }
                                }
                            };
                            args.push(arg);
                            match self.peek() {
                                Ok(Token::Comma) => { self.advance().ok(); }
                                Ok(Token::Gt) | Ok(Token::Shr) => { self.expect_gt()?; break; }
                                Ok(Token::Assign) => {
                                    self.advance().ok();
                                    let _eq_ty = self.parse_type()?;
                                    self.expect_gt()?;
                                    break;
                                }
                                _ => return Err(Diagnostic::error("expected ',' or '>' in type parameters")
                                    .with_code_str("E004")
                                    .with_help("generic type parameters use `<T, U>` syntax — separate with `,` and close with `>`")
                                    .with_suggestion("add `>` to close the generic type, or `,` to add another type argument")
                                    .with_span(self.span())),
                            }
                        }
                        let end = self.span().end;
                        Ok(Type::Generic(Self::alloc_shared(self.arena, Type::Path(smallvec::SmallVec::from(path), Span::new(start, end))), args, Span::new(start, end)))
                    } else {
                        let end = self.span().end;
                        Ok(Type::Path(smallvec::SmallVec::from(path), Span::new(start, end)))
                    }
                }
                Ok(Token::LParen) => {
                    let params = if matches!(self.peek(), Ok(Token::RParen)) {
                        self.advance().ok();
                        Vec::new()
                    } else {
                        let mut types = Vec::new();
                        loop {
                            types.push(self.parse_type()?);
                            match self.peek() {
                                Ok(Token::Comma) => { self.advance().ok(); }
                                Ok(Token::RParen) => { self.advance().ok(); break; }
                                _ => return Err(Diagnostic::error("expected ',' or ')' in tuple type")
                                    .with_code_str("E004")
                                    .with_help("tuple types use `(T, U)` syntax — separate with `,` and close with `)`")
                                    .with_suggestion("add `)` to close the tuple type, or `,` to add another element")
                                    .with_span(self.span())),
                            }
                        }
                        types
                    };
                    if matches!(self.peek(), Ok(Token::Arrow)) {
                        self.advance().ok();
                        let ret = Self::alloc_shared(self.arena, self.parse_type()?);
                        let end = self.span().end;
                        Ok(Type::Function { params, ret, span: Span::new(start, end) })
                    } else {
                        let end = self.span().end;
                        Ok(Type::Tuple(params, Span::new(start, end)))
                    }
                }
                Ok(Token::Bang) => {
                    let end = self.span().end;
                    Ok(Type::Never(Span::new(start, end)))
                }
                Ok(Token::Apostrophe) => {
                    match self.advance() {
                        Ok(Token::Ident(name)) => {
                            let end = self.span().end;
                            Ok(Type::Path(smallvec![Symbol::intern(&format!("'{}", name))], Span::new(start, end)))
                        }
                        _ => Err(Diagnostic::error("expected lifetime name after `'`")
                            .with_code_str("E004")
                            .with_span(self.span())),
                    }
                }
                Ok(tok) => Err(Diagnostic::error(format!("expected type, found {:?}", tok))
                    .with_code_str("E004")
                    .with_help("expected a valid type expression — try `Int<32>`, `&T`, `[T]`, `(A, B)`, etc.")
                    .with_span(self.span())),
                Err(()) => Err(Diagnostic::error("unexpected end of file in type")
                    .with_code_str("E002")
                    .with_help("type expression is incomplete — check for missing type arguments or brackets")
                    .with_suggestion("check for unclosed `<`, `[`, `(`, or `&` in the type expression")
                    .with_span(self.span())),
            },
        }
    }
}
