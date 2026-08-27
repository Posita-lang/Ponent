//! Parsing of attributes: `@name`, `@name(...)`, and `@name = value`.

use super::*;
use crate::ast::*;
use crate::diagnostics::Diagnostic;
use crate::lexer::Token;
use crate::symbol::Symbol;

impl<'input> Parser<'input> {
    // -------------------------------------------------------------------------
    // Attribute parser
    // -------------------------------------------------------------------------

    pub(super) fn parse_attribute(&mut self) -> Result<Attribute<'input>, Diagnostic> {
        let start = self.span().start;
        self.advance().ok();
        let name = match self.advance() {
            Ok(Token::Ident(name)) => name,
            Ok(tok) => self.keyword_to_ident(&tok).unwrap_or_else(|| Symbol::intern(&format!("{:?}", tok))),
            Err(()) => return Err(Diagnostic::error("unexpected end of file in attribute")
                .with_code_str("E002")
                .with_help("attributes must have a name — e.g. `@deprecated` or `@cfg(...)`")
                .with_suggestion("add an attribute name after `@`, or remove the `@` if this is not an attribute")
                .with_span(self.span())),
        };
        let mut args = Vec::new();
        let mut named_args = Vec::new();
        if matches!(self.peek(), Ok(Token::LParen)) {
            self.advance().ok();
            loop {
                if matches!(self.peek(), Ok(Token::RParen)) {
                    self.advance().ok();
                    break;
                }
                let is_named = match self.peek() {
                    Ok(Token::Ident(_)) => matches!(self.peek_next(), Some(Token::Assign)),
                    _ => false,
                };
                if is_named {
                    if let Ok(Token::Ident(key)) = self.advance() {
                        self.expect(Token::Assign)?;
                        let value = self.parse_expr()?;
                        named_args.push((key, value));
                    }
                } else {
                    args.push(self.parse_expr()?);
                }
                if matches!(self.peek(), Ok(Token::Comma)) {
                    self.advance().ok();
                } else {
                    self.expect(Token::RParen)?;
                    break;
                }
            }
        } else if matches!(self.peek(), Ok(Token::Assign)) {
            self.advance().ok();
            args.push(self.parse_expr()?);
        }
        let end = self.span().end;
        Ok(Attribute {
            name,
            args,
            named_args,
            span: Span::new(start, end),
        })
    }
}
