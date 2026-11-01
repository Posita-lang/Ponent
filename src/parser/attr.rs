//! Parsing of attributes: `@name`, `@name(...)`, and `@name = value`.

use super::*;
use crate::ast::*;
use crate::diagnostics::Diagnostic;
use crate::diagnostics::ErrCode;
use crate::lexer::Token;
use crate::symbol::Symbol;

impl<'src, 'input> Parser<'src, 'input> {
    // -------------------------------------------------------------------------
    // Attribute parser
    // -------------------------------------------------------------------------

    pub(super) fn parse_attribute(
        &mut self,
        bump: &'input bumpalo::Bump,
    ) -> Result<Attribute<'input>, Diagnostic<'src>> {
        let start = self.span().start;
        self.advance();
        let name = match self.advance() {
            Some(Token::Ident(name)) => name,
            Some(tok) => self.keyword_to_ident(&tok).unwrap_or_else(|| Symbol::intern(&format!("{:?}", tok))),
            None => return Err(Diagnostic::error("unexpected end of file in attribute")
                .with_code_str(ErrCode::E002)
                .with_help("attributes must have a name — e.g. `@deprecated` or `@cfg(...)`")
                .with_suggestion("add an attribute name after `@`, or remove the `@` if this is not an attribute")
                .with_span(self.span())),
        };
        let mut args = Vec::new();
        let mut named_args = Vec::new();
        if matches!(self.peek(), Some(Token::LParen)) {
            self.advance();
            loop {
                if matches!(self.peek(), Some(Token::RParen)) {
                    self.advance();
                    break;
                }
                let is_named = matches!(self.peek(), Some(Token::Ident(_)))
                    && matches!(self.peek_next(), Some(Token::Assign));
                if is_named {
                    if let Some(Token::Ident(key)) = self.advance() {
                        self.expect(Token::Assign)?;
                        let value = self.parse_expr(bump)?;
                        named_args.push((key, value));
                    }
                } else {
                    args.push(self.parse_expr(bump)?);
                }
                if matches!(self.peek(), Some(Token::Comma)) {
                    self.advance();
                } else {
                    self.expect(Token::RParen)?;
                    break;
                }
            }
        } else if matches!(self.peek(), Some(Token::Assign)) {
            self.advance();
            args.push(self.parse_expr(bump)?);
        }
        let end = self.span().end;
        Ok(Attribute {
            name,
            args,
            named_args,
            span: Span::new(self.file_id, start, end),
        })
    }
}
