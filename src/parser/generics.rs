//! Parsing of generic parameters (`<T, U, const N: usize>`), const arguments, and GAT lifetimes.

use super::*;
use crate::ast::*;
use crate::diagnostics::Diagnostic;
use crate::diagnostics::ErrCode;
use crate::lexer::Token;
use crate::symbol::Symbol;

impl<'src, 'input> Parser<'src, 'input> {
    // -------------------------------------------------------------------------
    // Type parameter list (including const parameters)
    // -------------------------------------------------------------------------

    pub(super) fn parse_type_params(
        &mut self,
        bump: &'input bumpalo::Bump,
    ) -> Result<Vec<TypeParam<'input>>, Diagnostic<'src>> {
        self.advance(); // consume <
        let mut p = Vec::new();
        loop {
            if matches!(self.peek(), Some(Token::Const)) {
                p.push(self.parse_const_param(bump)?);
            } else {
                let (name, is_lifetime) = if matches!(self.peek(), Some(Token::Apostrophe)) {
                    self.advance();
                    match self.advance() {
                        Some(Token::Ident(name)) => {
                            (Symbol::intern_prefixed('\'', name.as_str()), true)
                        }
                        _ => {
                            return Err(Diagnostic::error("expected lifetime name after `'`")
                                .with_code_str(ErrCode::E004)
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
                        Some(Token::Ident(name)) => (name, false),
                        _ => return Err(Diagnostic::error("expected type parameter name")
                            .with_code_str(ErrCode::E004)
                            .with_help("type parameters must have a name — e.g. `<T>` or `<K, V>`")
                            .with_suggestion(
                                "use a valid identifier like `T` or `Item` for the type parameter",
                            )
                            .with_span(self.span())),
                    }
                };
                let mut bounds = Vec::new();
                if !is_lifetime && matches!(self.peek(), Some(Token::Colon)) {
                    self.advance();
                    loop {
                        bounds.push(self.parse_type(bump)?);
                        if !matches!(self.peek(), Some(Token::Plus)) {
                            break;
                        }
                        self.advance();
                    }
                }
                let kind = if is_lifetime {
                    TypeParamKind::Lifetime
                } else {
                    TypeParamKind::Type
                };
                let span = self.span();
                p.push(TypeParam {
                    name,
                    bounds: bump.alloc_slice_clone(&bounds),
                    kind,
                    span,
                });
            }
            match self.peek() {
                Some(Token::Comma) => { self.advance(); }
                Some(Token::Gt) | Some(Token::Shr) => { self.expect_gt()?; break; }
                _ => return Err(Diagnostic::error("expected ',' or '>'")
                    .with_code_str(ErrCode::E004)
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

    pub(super) fn parse_const_param(
        &mut self,
        bump: &'input bumpalo::Bump,
    ) -> Result<TypeParam<'input>, Diagnostic<'src>> {
        let start = self.span().start;
        self.advance(); // consume `const`
        let name = match self.advance() {
            Some(Token::Ident(name)) => name,
            Some(tok) => {
                return Err(Diagnostic::error(format!(
                    "expected parameter name after `const`, found {:?}",
                    tok
                ))
                .with_code_str(ErrCode::E004)
                .with_help("`const` generic parameters require a name — e.g. `const N: usize`")
                .with_suggestion("add a parameter name after `const`, like `const N: usize`")
                .with_span(self.span()));
            }
            None => {
                return Err(Diagnostic::error("expected parameter name after `const`")
                    .with_code_str(ErrCode::E002)
                    .with_help("`const` generic parameters require a name — e.g. `const N: usize`")
                    .with_suggestion("add a parameter name after `const`, like `const N: usize`")
                    .with_span(self.span()));
            }
        };
        self.expect(Token::Colon)?;
        let ty = self.parse_type(bump)?;
        let default = if matches!(self.peek(), Some(Token::Assign)) {
            self.advance();
            Some(Self::alloc_shared(bump, self.parse_expr(bump)?))
        } else {
            None
        };
        let end = self.span().end;
        Ok(TypeParam {
            name,
            bounds: &[],
            kind: TypeParamKind::Const { ty, default },
            span: Span::new(self.file_id, start, end),
        })
    }

    // -------------------------------------------------------------------------
    // GAT lifetime parameters: `type Item<'a, 'b> = ...`
    // -------------------------------------------------------------------------

    pub(super) fn parse_associated_type_lifetime_params(
        &mut self,
    ) -> Result<Vec<Symbol>, Diagnostic<'src>> {
        if !matches!(self.peek(), Some(Token::Lt)) {
            return Ok(Vec::new());
        }
        self.advance();
        let mut params = Vec::new();
        loop {
            self.expect(Token::Apostrophe)?;
            let name = match self.advance() {
                Some(Token::Ident(n)) => n,
                _ => {
                    return Err(Diagnostic::error(
                        "expected a lifetime name after `'` in GAT parameters",
                    )
                    .with_code_str(ErrCode::E004)
                    .with_help("GAT lifetime parameters look like `type Item<'a>;`")
                    .with_span(self.span()));
                }
            };
            params.push(name);
            if matches!(self.peek(), Some(Token::Comma)) {
                self.advance();
                continue;
            }
            break;
        }
        self.expect_gt()?;
        Ok(params)
    }
}
