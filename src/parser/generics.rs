//! Parsing of generic parameters (`<T, U, const N: usize>`), const arguments, and GAT lifetimes.

use super::*;
use crate::ast::*;
use crate::diagnostics::Diagnostic;
use crate::lexer::Token;
use crate::symbol::Symbol;

impl<'input> Parser<'input> {
    // -------------------------------------------------------------------------
    // Type parameter list (including const parameters)
    // -------------------------------------------------------------------------

    pub(super) fn parse_type_params(&mut self) -> Result<Vec<TypeParam<'input>>, Diagnostic> {
        self.advance().ok(); // consume <
        let mut p = Vec::new();
        loop {
            if matches!(self.peek(), Ok(Token::Const)) {
                p.push(self.parse_const_param()?);
            } else {
                let (name, is_lifetime) = if matches!(self.peek(), Ok(Token::Apostrophe)) {
                    self.advance().ok();
                    match self.advance() {
                        Ok(Token::Ident(name)) => (Symbol::intern(&format!("'{}", name)), true),
                        _ => {
                            return Err(Diagnostic::error("expected lifetime name after `'`")
                                .with_code_str("E004")
                                .with_help(
                                    "lifetime parameters use `'name` syntax — e.g. `<'a, 'b, T>`",
                                )
                                .with_suggestion(
                                    "add a lifetime name after `'`, e.g. `<'a>` or `<'a, T>`",
                                )
                                .with_span(self.span()));
                        }
                    }
                } else {
                    match self.advance() {
                        Ok(Token::Ident(name)) => (name, false),
                        _ => return Err(Diagnostic::error("expected type parameter name")
                            .with_code_str("E004")
                            .with_help("type parameters must have a name — e.g. `<T>` or `<K, V>`")
                            .with_suggestion(
                                "use a valid identifier like `T` or `Item` for the type parameter",
                            )
                            .with_span(self.span())),
                    }
                };
                let mut bounds = Vec::new();
                if !is_lifetime && matches!(self.peek(), Ok(Token::Colon)) {
                    self.advance().ok();
                    loop {
                        bounds.push(self.parse_type()?);
                        if !matches!(self.peek(), Ok(Token::Plus)) {
                            break;
                        }
                        self.advance().ok();
                    }
                }
                let kind = if is_lifetime {
                    TypeParamKind::Lifetime
                } else {
                    TypeParamKind::Type
                };
                p.push(TypeParam {
                    name,
                    bounds,
                    kind,
                    span: Span::new(self.span().start, self.span().end),
                });
            }
            match self.peek() {
                Ok(Token::Comma) => { self.advance().ok(); }
                Ok(Token::Gt) | Ok(Token::Shr) => { self.expect_gt()?; break; }
                _ => return Err(Diagnostic::error("expected ',' or '>'")
                    .with_code_str("E004")
                    .with_help("type parameter lists use `<T, U>` syntax — separate parameters with `,` and close with `>`")
                    .with_suggestion("add `>` to close the type parameter list, or `,` to add another parameter")
                    .with_span(self.span())),
            }
        }
        Ok(p)
    }

    // -------------------------------------------------------------------------
    // Const generic parameter: `const N: Type = default`
    // -------------------------------------------------------------------------

    pub(super) fn parse_const_param(&mut self) -> Result<TypeParam<'input>, Diagnostic> {
        let start = self.span().start;
        self.advance().ok(); // consume `const`
        let name = match self.advance() {
            Ok(Token::Ident(name)) => name,
            Ok(tok) => {
                return Err(Diagnostic::error(format!(
                    "expected parameter name after `const`, found {:?}",
                    tok
                ))
                .with_code_str("E004")
                .with_help("`const` generic parameters require a name — e.g. `const N: usize`")
                .with_suggestion("add a parameter name after `const`, like `const N: usize`")
                .with_span(self.span()));
            }
            Err(()) => {
                return Err(Diagnostic::error("expected parameter name after `const`")
                    .with_code_str("E002")
                    .with_help("`const` generic parameters require a name — e.g. `const N: usize`")
                    .with_suggestion("add a parameter name after `const`, like `const N: usize`")
                    .with_span(self.span()));
            }
        };
        self.expect(Token::Colon)?;
        let ty = self.parse_type()?;
        let default = if matches!(self.peek(), Ok(Token::Assign)) {
            self.advance().ok();
            Some(Self::alloc_shared(self.arena, self.parse_expr()?))
        } else {
            None
        };
        let end = self.span().end;
        Ok(TypeParam {
            name,
            bounds: Vec::new(),
            kind: TypeParamKind::Const { ty, default },
            span: Span::new(start, end),
        })
    }

    // -------------------------------------------------------------------------
    // GAT lifetime parameters: `type Item<'a, 'b> = ...`
    // -------------------------------------------------------------------------

    pub(super) fn parse_associated_type_lifetime_params(
        &mut self,
    ) -> Result<Vec<Symbol>, Diagnostic> {
        if !matches!(self.peek(), Ok(Token::Lt)) {
            return Ok(Vec::new());
        }
        self.advance().ok();
        let mut params = Vec::new();
        loop {
            self.expect(Token::Apostrophe)?;
            let name = match self.advance() {
                Ok(Token::Ident(n)) => n,
                _ => {
                    return Err(Diagnostic::error(
                        "expected a lifetime name after `'` in GAT parameters",
                    )
                    .with_code_str("E004")
                    .with_help("GAT lifetime parameters look like `type Item<'a>;`")
                    .with_span(self.span()));
                }
            };
            params.push(name);
            if matches!(self.peek(), Ok(Token::Comma)) {
                self.advance().ok();
                continue;
            }
            break;
        }
        self.expect_gt()?;
        Ok(params)
    }
}
