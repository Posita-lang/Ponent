//! Parsing of patterns: literals, identifiers, structs, enums, and or‑patterns.

use super::*;
use crate::ast::*;
use crate::diagnostics::Diagnostic;
use crate::lexer::Token;
use crate::symbol::Symbol;

impl<'src, 'input> Parser<'src, 'input> {
    // -------------------------------------------------------------------------
    // Pattern entry
    // -------------------------------------------------------------------------

    pub(super) fn parse_pattern(&mut self, bump: &'input bumpalo::Bump) -> Result<Pattern<'input>, Diagnostic<'src>> {
        self.recursion_depth += 1;
        if self.recursion_depth > self.max_recursion_depth {
            self.recursion_depth -= 1;
            return Err(Diagnostic::error(format!("maximum recursion depth {} exceeded", self.max_recursion_depth))
                .with_code_str("E006")
                .with_help("the parser reached its recursion limit — the type/structure may be deeply nested or self-referential")
                .with_suggestion("try breaking up deeply nested structures, or use less complex type expressions")
                .with_span(self.span()));
        }
        let result = self.parse_pattern_inner(bump);
        self.recursion_depth -= 1;
        result
    }

    // -------------------------------------------------------------------------
    // Or‑pattern (used in match arms and catch)
    // -------------------------------------------------------------------------

    pub(super) fn parse_or_pattern(&mut self, bump: &'input bumpalo::Bump) -> Result<Pattern<'input>, Diagnostic<'src>> {
        let start = self.span().start;
        let first = self.parse_pattern(bump)?;
        if matches!(self.peek(), Ok(Token::Pipe)) {
            let mut patterns = bumpalo::collections::Vec::with_capacity_in(4, bump);
            patterns.push(first);
            while matches!(self.peek(), Ok(Token::Pipe)) {
                self.advance().ok();
                patterns.push(self.parse_pattern(bump)?);
            }
            Ok(Pattern::Or(patterns.into_bump_slice(), Span::new(self.file_id, start, self.span().end)))
        } else {
            Ok(first)
        }
    }

    // -------------------------------------------------------------------------
    // Inner pattern parser
    // -------------------------------------------------------------------------

    fn parse_pattern_inner(&mut self, bump: &'input bumpalo::Bump) -> Result<Pattern<'input>, Diagnostic<'src>> {
        let start = self.span().start;
        let tok = match self.peek() {
            Ok(t) => t.clone(),
            Err(()) => return Err(Diagnostic::error("unexpected end of file in pattern")
                .with_code_str("E002")
                .with_help("pattern is incomplete — expected a pattern expression (literal, variable, `_`, etc.)")
                .with_suggestion("add a pattern like `x`, `_`, `42`, or `Some(val)`")
                .with_span(self.span())),
        };
        match tok {
            Token::IntLiteral(_) | Token::FloatLiteral(_) | Token::StringLiteral(_)
            | Token::ByteStringLiteral(_) | Token::CharLiteral(_)
            | Token::ByteCharLiteral(_) | Token::True | Token::False => {
                let lit = self.parse_literal()?;
                Ok(Pattern::Literal(Self::alloc_shared(bump, lit), Span::new(self.file_id, start, self.span().end)))
            }
            Token::Ident(s) if s.eq_str("_") => {
                self.advance().ok();
                Ok(Pattern::Wildcard(Span::new(self.file_id, start, self.span().end)))
            }
            Token::Ident(_) => {
                let name = match self.advance() {
                    Ok(Token::Ident(n)) => n,
                    _ => unreachable!(),
                };
                if matches!(self.peek(), Ok(Token::LBrace)) {
                    let (fields, rest, _) = self.parse_struct_pattern_fields(bump)?;
                    let end = self.span().end;
                    Ok(Pattern::Struct { path: vec![name], fields, rest, span: Span::new(self.file_id, start, end) })
                } else if matches!(self.peek(), Ok(Token::LParen)) {
                    self.advance().ok();
                    let inner = self.parse_pattern(bump)?;
                    if matches!(self.peek(), Ok(Token::Colon)) {
                        self.advance().ok();
                        let ty = self.parse_type(bump)?;
                        self.expect(Token::RParen)?;
                        let end = self.span().end;
                        let annotated = Pattern::Annotated {
                            inner: Self::alloc_shared(bump, inner),
                            ty: Self::alloc_shared(bump, ty),
                            span: Span::new(self.file_id, start, end),
                        };
                        Ok(Pattern::Enum { path: vec![], variant: name, inner: Some(Self::alloc_shared(bump, annotated)), span: Span::new(self.file_id, start, end) })
                    } else {
                        self.expect(Token::RParen)?;
                        let end = self.span().end;
                        Ok(Pattern::Enum { path: vec![], variant: name, inner: Some(Self::alloc_shared(bump, inner)), span: Span::new(self.file_id, start, end) })
                    }
                } else if matches!(self.peek(), Ok(Token::ColonColon)) {
                    let mut path = vec![name];
                    self.advance().ok();
                    let variant_sym = match self.advance() {
                        Ok(tok) if tok.as_ident_symbol().is_some() => tok.as_ident_symbol().expect("guarded by is_some() above"),
                        _ => return Err(Diagnostic::error("expected variant name")
                            .with_code_str("E004")
                            .with_help("expected a variant name after `::` in enum pattern")
                            .with_suggestion("add a variant name after `::`, e.g. `Option::Some(val)`")
                            .with_span(self.span())),
                    };
                    if matches!(self.peek(), Ok(Token::LBrace)) {
                        let (fields, rest, _) = self.parse_struct_pattern_fields(bump)?;
                        path.push(variant_sym);
                        let end = self.span().end;
                        Ok(Pattern::Struct { path, fields, rest, span: Span::new(self.file_id, start, end) })
                    } else if matches!(self.peek(), Ok(Token::LParen)) {
                        self.advance().ok();
                        let p = self.parse_pattern(bump)?;
                        if matches!(self.peek(), Ok(Token::Colon)) {
                            self.advance().ok();
                            let ty = self.parse_type(bump)?;
                            self.expect(Token::RParen)?;
                            let end = self.span().end;
                            let annotated = Pattern::Annotated {
                                inner: Self::alloc_shared(bump, p),
                                ty: Self::alloc_shared(bump, ty),
                                span: Span::new(self.file_id, start, end),
                            };
                            Ok(Pattern::Enum { path, variant: variant_sym, inner: Some(Self::alloc_shared(bump, annotated)), span: Span::new(self.file_id, start, end) })
                        } else {
                            self.expect(Token::RParen)?;
                            let end = self.span().end;
                            Ok(Pattern::Enum { path, variant: variant_sym, inner: Some(Self::alloc_shared(bump, p)), span: Span::new(self.file_id, start, end) })
                        }
                    } else {
                        let end = self.span().end;
                        Ok(Pattern::Enum { path, variant: variant_sym, inner: None, span: Span::new(self.file_id, start, end) })
                    }
                } else {
                    Ok(Pattern::Ident(name, Span::new(self.file_id, start, self.span().end)))
                }
            }
            Token::LParen => {
                self.advance().ok();
                let first = self.parse_pattern(bump)?;
                if matches!(self.peek(), Ok(Token::Colon)) {
                    self.advance().ok();
                    let ty = self.parse_type(bump)?;
                    self.expect(Token::RParen)?;
                    let end = self.span().end;
                    Ok(Pattern::Annotated {
                        inner: Self::alloc_shared(bump, first),
                        ty: Self::alloc_shared(bump, ty),
                        span: Span::new(self.file_id, start, end),
                    })
                } else if matches!(self.peek(), Ok(Token::Comma)) {
                    let mut patterns = bumpalo::collections::Vec::with_capacity_in(4, bump);
                    patterns.push(first);
                    while matches!(self.peek(), Ok(Token::Comma)) {
                        self.advance().ok();
                        patterns.push(self.parse_pattern(bump)?);
                    }
                    self.expect(Token::RParen)?;
                    Ok(Pattern::Tuple(patterns.into_bump_slice(), Span::new(self.file_id, start, self.span().end)))
                } else {
                    self.expect(Token::RParen)?;
                    Ok(Pattern::Tuple(Self::alloc_slice(bump, std::slice::from_ref(&first)), Span::new(self.file_id, start, self.span().end)))
                }
            }
            _ => Err(Diagnostic::error("expected pattern")
                .with_code_str("E004")
                .with_help("expected a valid pattern — try a literal, variable name, `_`, struct pattern, or tuple pattern")
                .with_suggestion("try `x`, `_`, `42`, `true`, `Point { x, y }`, or `Some(val)`")
                .with_span(self.span())),
        }
    }

    // -------------------------------------------------------------------------
    // Struct pattern fields: `{ field1, field2: pat, .. }`
    // -------------------------------------------------------------------------

    pub(super) fn parse_struct_pattern_fields(
        &mut self,
        bump: &'input bumpalo::Bump,
    ) -> Result<(&'input [(Symbol, Pattern<'input>)], bool, Span), Diagnostic<'src>> {
        let start = self.span().start;
        self.expect(Token::LBrace)?;
        let mut fields = bumpalo::collections::Vec::with_capacity_in(4, bump);
        let mut rest = false;
        loop {
            if matches!(self.peek(), Ok(Token::RBrace)) {
                self.advance().ok();
                break;
            }
            if matches!(self.peek(), Ok(Token::DotDot)) {
                self.advance().ok();
                rest = true;
                if matches!(self.peek(), Ok(Token::Comma)) {
                    self.advance().ok();
                }
                continue;
            }
            let field_tok = self.advance();
            match field_tok {
                Ok(tok) if tok.as_ident_symbol().is_some() => {
                    let field_name = tok.as_ident_symbol().expect("guarded by is_some() above");
                    let field_pattern = if matches!(self.peek(), Ok(Token::Colon)) {
                        self.advance().ok();
                        self.parse_pattern(bump)?
                    } else {
                        Pattern::Ident(field_name, self.span())
                    };
                    fields.push((field_name, field_pattern));
                }
                _ => return Err(Diagnostic::error("expected field name or `..`")
                    .with_code_str("E004")
                    .with_help("pattern fields must have a name — e.g. `Point { x, y }`, or use `..` for remaining fields")
                    .with_suggestion("add a field name like `x`, `name`, or `value`, or add `..` to ignore the rest")
                    .with_span(self.span())),
            }
            if matches!(self.peek(), Ok(Token::Comma)) {
                self.advance().ok();
            } else {
                self.expect(Token::RBrace)?;
                break;
            }
        }
        let end = self.span().end;
        Ok((fields.into_bump_slice(), rest, Span::new(self.file_id, start, end)))
    }
}
