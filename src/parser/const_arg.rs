//! Parsing of const generic arguments: detection and simple‑expression validation.

use super::*;
use crate::ast::*;
use crate::diagnostics::Diagnostic;
use crate::lexer::Token;

impl<'input> Parser<'input> {
    /// Rustc-style pre-check: does the current token definitively start a
    /// const generic argument rather than a type? When true, the generic arg
    /// loop parses directly as an expression without any type-first attempt.
    /// Analogous to rustc's `can_begin_const_arg`/`check_const_arg`.
    pub(super) fn check_const_arg(&mut self) -> bool {
        match self.peek() {
            Ok(Token::True)
            | Ok(Token::False)
            | Ok(Token::CharLiteral(_))
            | Ok(Token::StringLiteral(_))
            | Ok(Token::ByteStringLiteral(_))
            | Ok(Token::FloatLiteral(_))
            | Ok(Token::Minus)
            | Ok(Token::LBrace)
            | Ok(Token::If)
            | Ok(Token::Match) => true,
            Ok(Token::Ident(_)) => matches!(self.peek_next(), Some(Token::LParen)),
            _ => false,
        }
    }

    /// Parse a const generic argument (value side), e.g. `<{ 2 + 2 }>`, `<N>`, or `{ 42 }`.
    /// Returns an `AnonConst<'input>` wrapping the expression.
    /// Per the Posita syntax spec: complex expressions MUST be wrapped in `{ }`;
    /// only simple literals and identifier paths may appear unbraced.
    /// Analogous to rustc's `parse_const_arg` (rustc_parse/src/parser/path.rs).
    pub(super) fn parse_const_arg(&mut self) -> Result<AnonConst<'input>, Diagnostic> {
        let start = self.span().start;
        let value = if matches!(self.peek(), Ok(Token::LBrace)) {
            self.advance().ok();
            let body = self.parse_block()?;
            self.expect(Token::RBrace)?;
            let end = self.span().end;
            Expr::Block(body, Span::new(start, end))
        } else {
            let expr =
                self.with_restrictions(ParseRestrictions::NO_COMPARISON, |this| this.parse_expr())?;
            if !Self::is_simple_const_expr(&expr) {
                let span = expr.span();
                let end = self.span().end;
                return Err(Diagnostic::error(
                    "complex const generic argument must be wrapped in `{ }`",
                )
                .with_code_str("E004")
                .with_help(
                    "wrap the expression in braces — e.g. `<{ N + 1 }>` instead of `<N + 1>`",
                )
                .with_suggestion("add `{` before and `}` after the expression")
                .with_span(Span::new(start, end)));
            }
            expr
        };
        let end = self.span().end;
        Ok(AnonConst {
            value: Self::alloc_shared(self.arena, value),
            span: Span::new(start, end),
        })
    }

    /// Check whether an unbraced const generic argument expression is simple enough
    /// to be unambiguous — a literal, a single-segment identifier path, or a unary
    /// `-`/`+` applied to one of those. Complex expressions like `N + 1` must be
    /// wrapped in `{ }`.
    pub(super) fn is_simple_const_expr(expr: &Expr<'input>) -> bool {
        match expr {
            Expr::Literal(..) => true,
            Expr::Ident(..) => true,
            Expr::Path(path, _) if path.len() <= 1 => true,
            Expr::UnaryOp {
                op, expr: inner, ..
            } if matches!(op, UnaryOp::Neg) => Self::is_simple_const_expr(inner),
            Expr::Call { comptime: true, .. } => true,
            _ => false,
        }
    }
}
