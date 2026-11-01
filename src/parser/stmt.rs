//! Parsing of statements: blocks, variable definitions, control flow, scope_cleanup, etc.

use super::*;
use crate::ast::*;
use crate::diagnostics::Diagnostic;
use crate::diagnostics::ErrCode;
use crate::lexer::Token;
use crate::symbol::Symbol;

impl<'src, 'input> Parser<'src, 'input> {
    // -------------------------------------------------------------------------
    // Block parsing
    // -------------------------------------------------------------------------

    pub(super) fn parse_block(
        &mut self,
        bump: &'input bumpalo::Bump,
    ) -> Result<&'input [Stmt<'input>], Diagnostic<'src>> {
        self.recursion_depth += 1;
        if self.recursion_depth > self.max_recursion_depth {
            self.recursion_depth -= 1;
            return Err(Diagnostic::error(format!("maximum recursion depth {} exceeded", self.max_recursion_depth))
                .with_code_str(ErrCode::E006)
                .with_help("the parser reached its recursion limit — the type/structure may be deeply nested or self-referential")
                .with_suggestion("try breaking up deeply nested structures, or use less complex type expressions")
                .with_span(self.span()));
        }
        let result = self.parse_block_inner(bump);
        self.recursion_depth -= 1;
        result
    }

    fn parse_block_inner(
        &mut self,
        bump: &'input bumpalo::Bump,
    ) -> Result<&'input [Stmt<'input>], Diagnostic<'src>> {
        self.without_restrictions(ParseRestrictions::NO_STRUCT_LITERAL, |this| {
            // Build the statement list directly in the arena (zero-copy):
            // `into_bump_slice` hands out the buffer without a clone/drop.
            let mut stmts = bumpalo::collections::Vec::with_capacity_in(8, bump);
            loop {
                match this.peek() {
                    Some(Token::RBrace) | None => break,
                    _ => match this.parse_stmt(bump) {
                        Ok(stmt) => stmts.push(stmt),
                        Err(diag) => {
                            this.diagnostics.push(diag);
                            this.synchronize();
                            stmts.push(Stmt::Error(Span::new(
                                this.file_id,
                                this.span().start,
                                this.span().start,
                            )));
                        }
                    },
                }
            }
            Ok(stmts.into_bump_slice())
        })
    }

    // -------------------------------------------------------------------------
    // Statement dispatch
    // -------------------------------------------------------------------------

    pub(super) fn parse_stmt(
        &mut self,
        bump: &'input bumpalo::Bump,
    ) -> Result<Stmt<'input>, Diagnostic<'src>> {
        self.recursion_depth += 1;
        if self.recursion_depth > self.max_recursion_depth {
            self.recursion_depth -= 1;
            return Err(Diagnostic::error(format!("maximum recursion depth {} exceeded", self.max_recursion_depth))
                .with_code_str(ErrCode::E006)
                .with_help("the parser reached its recursion limit — the type/structure may be deeply nested or self-referential")
                .with_suggestion("try breaking up deeply nested structures, or use less complex type expressions")
                .with_span(self.span()));
        }
        let result = self.parse_stmt_inner(bump);
        self.recursion_depth -= 1;
        result
    }

    fn parse_stmt_inner(
        &mut self,
        bump: &'input bumpalo::Bump,
    ) -> Result<Stmt<'input>, Diagnostic<'src>> {
        // Collect leading @attributes — only valid before comptime inside
        // function bodies (e.g. @deprecated("use bar") comptime { ... }).
        // Use checkpoint/restore to avoid consuming `@` in expression-level
        // constructs like `@compile_error("msg")`.
        let mut attributes = Vec::new();
        if matches!(self.peek(), Some(Token::At)) {
            let cp = self.checkpoint();
            loop {
                match self.peek() {
                    Some(Token::At) => {
                        attributes.push(self.parse_attribute(bump)?);
                    }
                    _ => break,
                }
            }
            if !matches!(self.peek(), Some(Token::Comptime)) {
                // Not followed by comptime — restore cursor so the `@`
                // can be parsed as part of an expression (e.g. @compile_error).
                self.restore(cp);
                attributes = Vec::new();
            }
        }
        // A loop label prefix: `'label: while ...` / `'label: for ...` /
        // `'label: loop ...`.  `continue 'label;` and `leave 'label;`
        // (SYNTAX.md §Loops) target the enclosing loop with that label.
        // Use checkpoint/restore so a stray apostrophe (e.g. in a lifetime
        // position) falls through to the normal statement dispatch.
        let label = if matches!(self.peek(), Some(Token::Apostrophe)) {
            let cp = self.checkpoint();
            self.advance();
            match self.advance() {
                Some(Token::Ident(name)) if matches!(self.peek(), Some(Token::Colon)) => {
                    self.advance();
                    match self.peek() {
                        Some(Token::While) | Some(Token::For) | Some(Token::Loop) => Some(name),
                        _ => {
                            self.restore(cp);
                            None
                        }
                    }
                }
                _ => {
                    self.restore(cp);
                    None
                }
            }
        } else {
            None
        };
        match self.peek() {
            Some(Token::Set) | Some(Token::Let) => self.parse_variable_def(attributes, bump),
            Some(Token::If) => self.parse_if_stmt(bump),
            Some(Token::While) => self.parse_while_stmt(label, bump),
            Some(Token::For) => self.parse_for_stmt(label, bump),
            Some(Token::Loop) => self.parse_loop_stmt(label, bump),
            Some(Token::Leave) => self.parse_leave_stmt(bump),
            Some(Token::Continue) => self.parse_continue_stmt(),
            Some(Token::Return) => self.parse_return_stmt(bump),
            Some(Token::LBrace) => {
                let start = self.span().start;
                self.advance();
                let body = self.parse_block(bump)?;
                self.expect(Token::RBrace)?;
                let end = self.span().end;
                Ok(Stmt::Expression(Expr::Block(
                    body,
                    Span::new(self.file_id, start, end),
                )))
            }
            Some(Token::Comptime) => self.parse_comptime_block_stmt(attributes, bump),
            Some(Token::Generate) => self.parse_generate_stmt(bump),
            Some(Token::ScopeCleanup) => self.parse_scope_cleanup(bump),
            Some(Token::Trigger) => self.parse_trigger(),
            Some(Token::Unsafe) => self.parse_unsafe_block(bump),
            Some(Token::Ghost) => self.parse_ghost_variable(bump),
            Some(Token::Isolate) => self.parse_isolate_block(bump),
            Some(Token::Match) => {
                let start = self.span().start;
                let expr = self.parse_match_expr(bump)?;
                self.expect(Token::Semicolon)?;
                let end = self.span().end;
                Ok(Stmt::Expression(expr))
            }
            Some(Token::Def) => {
                // Nested function definitions: parse them as function
                // definitions (rustc-style — items in blocks are collected
                // and referenced, not rejected).
                self.advance();
                let item = self
                    .with_restrictions(ParseRestrictions::ALLOW_TYPE_PARAMS, |this| {
                        this.parse_function_def(Vec::new(), None, false, false, bump)
                    })?;
                Ok(Stmt::Item(Self::alloc_shared(bump, item)))
            }
            Some(Token::Trait) => {
                let item = self.parse_trait_def(attributes, None, bump)?;
                Ok(Stmt::Item(Self::alloc_shared(bump, item)))
            }
            Some(Token::Impl) => {
                let item = self
                    .with_restrictions(ParseRestrictions::ALLOW_TYPE_PARAMS, |this| {
                        this.parse_impl_block(attributes, bump)
                    })?;
                Ok(Stmt::Item(Self::alloc_shared(bump, item)))
            }
            Some(Token::Constraint) => {
                let item = self.parse_constraint(bump)?;
                Ok(Stmt::Item(Self::alloc_shared(bump, item)))
            }
            Some(Token::Type) => {
                // Nested type definitions — blocks may contain items
                // (rustc-style).  Before this arm existed, `type` fell
                // through to the expression fallback and produced a
                // misleading "expected expression" (E007) false negative;
                // the synchronize fix only prevented the infinite
                // recovery loop, not the error itself.
                //
                // NOTE: do NOT pre-consume `type` here — unlike
                // parse_function_def (which expects `def` already
                // consumed), parse_type_def consumes the `type` keyword
                // itself.
                let item = self
                    .with_restrictions(ParseRestrictions::ALLOW_TYPE_PARAMS, |this| {
                        this.parse_type_def(Vec::new(), None, bump)
                    })?;
                // The Alias / Opaque branches of parse_type_def consume an
                // optional trailing `;`, but the Struct / Enum branches do
                // not.  In statement position users habitually terminate
                // with `;` — swallow it here so the leftover semicolon
                // does not trigger a spurious error.
                if matches!(self.peek(), Some(Token::Semicolon)) {
                    self.advance();
                }
                Ok(Stmt::Item(Self::alloc_shared(bump, item)))
            }
            _ => {
                let start = self.span().start;
                let lhs = self.parse_expr(bump)?;
                if matches!(
                    self.peek(),
                    Some(Token::Assign)
                        | Some(Token::PlusEq)
                        | Some(Token::MinusEq)
                        | Some(Token::StarEq)
                        | Some(Token::SlashEq)
                        | Some(Token::ShlEq)
                        | Some(Token::ShrEq)
                ) {
                    let op_token = self.advance().ok_or_else(|| Diagnostic::error("unexpected token")
                        .with_code_str(ErrCode::E003)
                        .with_help("expected an assignment operator (`=`, `+=`, `-=`, `*=`, `/=`, `<<=`, `>>=`) after the target")
                        .with_span(Span::new(self.file_id, 0, 0)))?;
                    let op = match op_token {
                        Token::Assign => None,
                        Token::PlusEq => Some(BinOp::Add),
                        Token::MinusEq => Some(BinOp::Sub),
                        Token::StarEq => Some(BinOp::Mul),
                        Token::SlashEq => Some(BinOp::Div),
                        Token::ShlEq => Some(BinOp::Shl),
                        Token::ShrEq => Some(BinOp::Shr),
                        _ => unreachable!(),
                    };
                    let value = self.parse_expr(bump)?;
                    self.expect(Token::Semicolon)?;
                    let end = self.span().end;
                    // The lvalue invariant is established HERE (parse time):
                    // an invalid assignment target (`1 = 2`, `f() = x`, …)
                    // is a parse error with the target's own span, and
                    // downstream consumers of `Stmt::Assign` never re-check.
                    let target = LValue::try_from_expr(Self::alloc_shared(bump, lhs)).map_err(
                        |bad_span| {
                            Diagnostic::error("invalid assignment target")
                                .with_code_str(ErrCode::E003)
                                .with_span(bad_span)
                                .with_help(
                                    "only variables, paths, field/index accesses, and `*deref` can be assigned to",
                                )
                        },
                    )?;
                    Ok(Stmt::Assign {
                        target,
                        op,
                        value,
                        span: Span::new(self.file_id, start, end),
                    })
                } else {
                    let at_end = matches!(self.peek(), Some(Token::RBrace) | None);
                    if at_end {
                        Ok(Stmt::Expression(lhs))
                    } else {
                        if self.restrictions.contains(ParseRestrictions::STMT_EXPR) {
                            self.expect(Token::Semicolon)?;
                        }
                        Ok(Stmt::Expression(lhs))
                    }
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Variable definition (`set` / `let`)
    // -------------------------------------------------------------------------

    pub(super) fn parse_variable_def(
        &mut self,
        attributes: Vec<Attribute<'input>>,
        bump: &'input bumpalo::Bump,
    ) -> Result<Stmt<'input>, Diagnostic<'src>> {
        let start = self.span().start;
        let kind = match self.advance().ok_or_else(|| {
            Diagnostic::error("unexpected token")
                .with_code_str(ErrCode::E003)
                .with_help("expected `set` or `let` to begin a variable definition")
                .with_span(Span::new(self.file_id, 0, 0))
        })? {
            Token::Set => VariableKind::Set,
            Token::Let => VariableKind::Let,
            _ => unreachable!(),
        };
        let mutable = if kind == VariableKind::Set && matches!(self.peek(), Some(Token::Mut)) {
            self.advance();
            true
        } else {
            false
        };
        let (name, pattern) = if kind == VariableKind::Let
            && matches!(
                self.peek(),
                Some(Token::LParen) | Some(Token::LBracket) | Some(Token::Ident(_))
            ) {
            let ident = if let Some(Token::Ident(s)) = self.peek() {
                Some(*s)
            } else {
                None
            };
            if let Some(s) = ident {
                let next_is_pattern = matches!(
                    self.peek_next(),
                    Some(Token::LBrace) | Some(Token::LParen) | Some(Token::ColonColon)
                );
                if !next_is_pattern
                    && (s.eq_str("_")
                        || s.as_str()
                            .chars()
                            .next()
                            .map_or(false, |c| c.is_alphabetic()))
                {
                    self.advance();
                    (Some(s), None)
                } else {
                    (None, Some(self.parse_pattern(bump)?))
                }
            } else {
                (None, Some(self.parse_pattern(bump)?))
            }
        } else {
            let ident = match self.advance() {
                Some(Token::Ident(name)) => name,
                Some(Token::Auto) => Symbol::intern("auto"),
                Some(tok) => return Err(Diagnostic::error(format!("expected variable name, found {:?}", tok))
                    .with_code_str(ErrCode::E004)
                    .with_help("a variable name must follow `set` or `let` — use a valid identifier")
                    .with_suggestion("use a valid identifier like `x` or `counter` for the variable name")
                    .with_span(self.span())),
                None => return Err(Diagnostic::error("unexpected end of file in variable definition")
                    .with_code_str(ErrCode::E002)
                    .with_help("variable definition is incomplete — expected a name or pattern after `set`/`let`")
                    .with_suggestion("add a variable name after `set`/`let`, e.g. `set x = 42;`")
                    .with_span(self.span())),
            };
            (Some(ident), None)
        };
        let ty = if matches!(self.peek(), Some(Token::Colon)) {
            self.advance();
            Some(self.parse_type(bump)?)
        } else {
            None
        };
        let type_modifiers = if matches!(self.peek(), Some(Token::With)) {
            self.parse_type_modifiers(bump)?
        } else {
            Vec::new()
        };
        let type_captures = if let Some(ref name_str) = name {
            if name_str.eq_str("auto") && matches!(self.peek(), Some(Token::Lt)) {
                self.parse_type_capture_params()?
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };
        let value = if matches!(self.peek(), Some(Token::Assign)) {
            self.advance();
            Some(self.parse_expr(bump)?)
        } else {
            None
        };
        let else_branch = if kind == VariableKind::Let && matches!(self.peek(), Some(Token::Else)) {
            self.advance();
            self.expect(Token::LBrace)?;
            let block = self.parse_block(bump)?;
            self.expect(Token::RBrace)?;
            Some(block)
        } else {
            None
        };
        self.expect(Token::Semicolon)?;
        let end = self.span().end;
        Ok(Stmt::VariableDef {
            kind,
            mutable,
            name,
            pattern,
            ty,
            value,
            else_branch,
            span: Span::new(self.file_id, start, end),
            attributes: bump.alloc_slice_clone(&attributes),
            doc: None,
            type_captures: bump.alloc_slice_clone(&type_captures),
            type_modifiers: bump.alloc_slice_clone(&type_modifiers),
        })
    }

    // -------------------------------------------------------------------------
    // `comptime` block as statement
    // -------------------------------------------------------------------------

    fn parse_comptime_block_stmt(
        &mut self,
        attributes: Vec<Attribute<'input>>,
        bump: &'input bumpalo::Bump,
    ) -> Result<Stmt<'input>, Diagnostic<'src>> {
        let start = self.span().start;
        self.advance();
        let trusted = if matches!(self.peek(), Some(Token::At)) {
            let attr = self.parse_attribute(bump)?;
            if !attr.name.eq_str("trusted") {
                return Err(
                    Diagnostic::error("only `@trusted` is allowed after `comptime`")
                        .with_code_str(ErrCode::E004)
                        .with_span(attr.span)
                        .with_help("use `comptime @trusted { ... }` for a trusted comptime block"),
                );
            }
            true
        } else {
            false
        };
        let captures: &'input [(Symbol, Span)] = if matches!(self.peek(), Some(Token::LBracket)) {
            let mut captures = bumpalo::collections::Vec::with_capacity_in(4, bump);
            self.advance();
            loop {
                let name_span = self.span();
                match self.peek() {
                    Some(Token::RBracket) => {
                        self.advance();
                        break;
                    }
                    Some(Token::Ident(name)) => {
                        captures.push((*name, name_span));
                        self.advance();
                        match self.peek() {
                            Some(Token::Comma) => {
                                self.advance();
                            }
                            Some(Token::RBracket) => continue,
                            other => {
                                return Err(Diagnostic::error(format!(
                                    "expected ',' or ']' after capture name, found {:?}",
                                    other
                                ))
                                .with_code_str(ErrCode::E005)
                                .with_span(self.span()));
                            }
                        }
                    }
                    other => {
                        return Err(Diagnostic::error(format!(
                            "expected capture name or ']', found {:?}",
                            other
                        ))
                        .with_code_str(ErrCode::E005)
                        .with_span(self.span()));
                    }
                }
            }
            captures.into_bump_slice()
        } else {
            &[]
        };
        let trusted = if matches!(self.peek(), Some(Token::At)) {
            let attr = self.parse_attribute(bump)?;
            if !attr.name.eq_str("trusted") {
                return Err(Diagnostic::error("only `@trusted` is allowed after `comptime [captures]`")
                    .with_code_str(ErrCode::E004)
                    .with_span(attr.span)
                    .with_help("use `comptime [captures] @trusted { ... }` for a trusted comptime block with captures")
                    .with_suggestion("remove the unknown attribute, or use `@trusted`"));
            }
            true
        } else {
            trusted
        };
        self.expect(Token::LBrace)?;
        let body = self.parse_block(bump)?;
        self.expect(Token::RBrace)?;
        let end = self.span().end;
        let item = Item::ComptimeBlock {
            captures,
            trusted,
            attributes: bump.alloc_slice_clone(&attributes),
            body,
            span: Span::new(self.file_id, start, end),
        };
        Ok(Stmt::Item(Self::alloc_shared(bump, item)))
    }

    // -------------------------------------------------------------------------
    // `generate` block as statement
    // -------------------------------------------------------------------------

    fn parse_generate_stmt(
        &mut self,
        bump: &'input bumpalo::Bump,
    ) -> Result<Stmt<'input>, Diagnostic<'src>> {
        let start = self.span().start;
        self.advance();
        match self.advance() {
            Some(Token::For) => {}
            Some(tok) => {
                return Err(Diagnostic::error(format!(
                    "expected `for` after `generate`, found {:?}",
                    tok
                ))
                .with_span(self.span()));
            }
            None => {
                return Err(
                    Diagnostic::error("expected `for` after `generate`").with_span(self.span())
                );
            }
        }
        let for_type = Self::alloc_shared(bump, self.parse_type(bump)?);
        self.expect(Token::LBrace)?;
        let body = self.parse_block(bump)?;
        self.expect(Token::RBrace)?;
        let end = self.span().end;
        let item = Item::Generate {
            attributes: &[],
            for_type,
            body,
            span: Span::new(self.file_id, start, end),
        };
        Ok(Stmt::Item(Self::alloc_shared(bump, item)))
    }

    // -------------------------------------------------------------------------
    // If / if‑let
    // -------------------------------------------------------------------------

    pub(super) fn parse_if_stmt(
        &mut self,
        bump: &'input bumpalo::Bump,
    ) -> Result<Stmt<'input>, Diagnostic<'src>> {
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
            let then_branch = self.parse_block(bump)?;
            self.expect(Token::RBrace)?;
            let else_branch = if matches!(self.peek(), Some(Token::Else)) {
                self.advance();
                if matches!(self.peek(), Some(Token::If)) {
                    Some(Self::alloc_slice(bump, &[self.parse_if_stmt(bump)?]))
                } else {
                    self.expect(Token::LBrace)?;
                    let block = self.parse_block(bump)?;
                    self.expect(Token::RBrace)?;
                    Some(block)
                }
            } else {
                None
            };
            let end = self.span().end;
            return Ok(Stmt::IfLet {
                pattern,
                scrutinee,
                then_branch,
                else_branch,
                span: Span::new(self.file_id, start, end),
            });
        }
        let cond = self.with_restrictions(ParseRestrictions::NO_STRUCT_LITERAL, |this| {
            this.parse_expr(bump)
        })?;
        self.expect(Token::LBrace)?;
        let then_branch = self.parse_block(bump)?;
        self.expect(Token::RBrace)?;
        let else_branch = if matches!(self.peek(), Some(Token::Else)) {
            self.advance();
            if matches!(self.peek(), Some(Token::If)) {
                Some(Self::alloc_slice(bump, &[self.parse_if_stmt(bump)?]))
            } else {
                self.expect(Token::LBrace)?;
                let block = self.parse_block(bump)?;
                self.expect(Token::RBrace)?;
                Some(block)
            }
        } else {
            None
        };
        let end = self.span().end;
        Ok(Stmt::If {
            cond,
            then_branch,
            else_branch,
            span: Span::new(self.file_id, start, end),
        })
    }

    // -------------------------------------------------------------------------
    // While / while‑let
    // -------------------------------------------------------------------------

    pub(super) fn parse_while_stmt(
        &mut self,
        label: Option<Symbol>,
        bump: &'input bumpalo::Bump,
    ) -> Result<Stmt<'input>, Diagnostic<'src>> {
        let start = self.span().start;
        self.advance();
        if matches!(self.peek(), Some(Token::Let)) {
            self.advance();
            let pattern = self.parse_or_pattern(bump)?;
            self.expect(Token::Assign)?;
            // RAII instead of nested closures: one guard covers the
            // scrutinee, every invariant/decreases expression, and restores
            // on scope exit (including `?` early returns below).
            let mut this = self.push_restrictions(ParseRestrictions::NO_STRUCT_LITERAL);
            let scrutinee = this.parse_expr(bump)?;
            let mut invariant = None;
            let mut decreases = None;
            while matches!(this.peek(), Some(Token::Invariant) | Some(Token::Decreases)) {
                match this.peek() {
                    Some(Token::Invariant) => {
                        this.advance();
                        invariant = Some(this.parse_expr(bump)?);
                    }
                    Some(Token::Decreases) => {
                        this.advance();
                        decreases = Some(this.parse_expr(bump)?);
                    }
                    _ => break,
                }
            }
            drop(this);
            self.expect(Token::LBrace)?;
            let body = self.parse_block(bump)?;
            self.expect(Token::RBrace)?;
            let end = self.span().end;
            return Ok(Stmt::WhileLet {
                label,
                pattern,
                scrutinee,
                body,
                invariant,
                decreases,
                span: Span::new(self.file_id, start, end),
            });
        }
        let cond = self.with_restrictions(ParseRestrictions::NO_STRUCT_LITERAL, |this| {
            this.parse_expr(bump)
        })?;
        let mut invariant = None;
        let mut decreases = None;
        while matches!(self.peek(), Some(Token::Invariant) | Some(Token::Decreases)) {
            match self.peek() {
                Some(Token::Invariant) => {
                    self.advance();
                    let inv = self
                        .with_restrictions(ParseRestrictions::NO_STRUCT_LITERAL, |this| {
                            this.parse_expr(bump)
                        })?;
                    invariant = Some(inv);
                }
                Some(Token::Decreases) => {
                    self.advance();
                    let dec = self
                        .with_restrictions(ParseRestrictions::NO_STRUCT_LITERAL, |this| {
                            this.parse_expr(bump)
                        })?;
                    decreases = Some(dec);
                }
                _ => break,
            }
        }
        self.expect(Token::LBrace)?;
        let body = self.parse_block(bump)?;
        self.expect(Token::RBrace)?;
        let end = self.span().end;
        Ok(Stmt::While {
            label,
            cond,
            body,
            invariant,
            decreases,
            span: Span::new(self.file_id, start, end),
        })
    }

    // -------------------------------------------------------------------------
    // For loop
    // -------------------------------------------------------------------------

    pub(super) fn parse_for_stmt(
        &mut self,
        label: Option<Symbol>,
        bump: &'input bumpalo::Bump,
    ) -> Result<Stmt<'input>, Diagnostic<'src>> {
        let start = self.span().start;
        self.advance();
        let pattern = self.parse_pattern(bump)?;
        self.expect(Token::In)?;
        let iterable = self.with_restrictions(ParseRestrictions::NO_STRUCT_LITERAL, |this| {
            this.parse_expr(bump)
        })?;
        let mut invariant = None;
        let mut decreases = None;
        while matches!(self.peek(), Some(Token::Invariant) | Some(Token::Decreases)) {
            match self.peek() {
                Some(Token::Invariant) => {
                    self.advance();
                    let inv = self
                        .with_restrictions(ParseRestrictions::NO_STRUCT_LITERAL, |this| {
                            this.parse_expr(bump)
                        })?;
                    invariant = Some(inv);
                }
                Some(Token::Decreases) => {
                    self.advance();
                    let dec = self
                        .with_restrictions(ParseRestrictions::NO_STRUCT_LITERAL, |this| {
                            this.parse_expr(bump)
                        })?;
                    decreases = Some(dec);
                }
                _ => break,
            }
        }
        self.expect(Token::LBrace)?;
        let body = self.parse_block(bump)?;
        self.expect(Token::RBrace)?;
        let end = self.span().end;
        Ok(Stmt::For {
            label,
            pattern,
            iterable,
            body,
            invariant,
            decreases,
            span: Span::new(self.file_id, start, end),
        })
    }

    // -------------------------------------------------------------------------
    // Loop
    // -------------------------------------------------------------------------

    pub(super) fn parse_loop_stmt(
        &mut self,
        label: Option<Symbol>,
        bump: &'input bumpalo::Bump,
    ) -> Result<Stmt<'input>, Diagnostic<'src>> {
        let start = self.span().start;
        self.advance();
        self.expect(Token::LBrace)?;
        let body = self.parse_block(bump)?;
        self.expect(Token::RBrace)?;
        let end = self.span().end;
        Ok(Stmt::Loop {
            label,
            body,
            span: Span::new(self.file_id, start, end),
        })
    }

    // -------------------------------------------------------------------------
    // Leave
    // -------------------------------------------------------------------------

    pub(super) fn parse_leave_stmt(
        &mut self,
        bump: &'input bumpalo::Bump,
    ) -> Result<Stmt<'input>, Diagnostic<'src>> {
        let start = self.span().start;
        self.advance();
        if matches!(self.peek(), Some(Token::With)) {
            self.advance();
            let expr = self.parse_expr(bump)?;
            self.expect(Token::Semicolon)?;
            let end = self.span().end;
            Ok(Stmt::Expression(Expr::LeaveWith {
                expr: Self::alloc_shared(bump, expr),
                is_return: false,
                span: Span::new(self.file_id, start, end),
            }))
        } else {
            let label = match self.peek().clone() {
                Some(Token::Apostrophe) => {
                    self.advance();
                    match self.advance() {
                        Some(Token::Ident(l)) => Some(l),
                        _ => {
                            return Err(Diagnostic::error("expected a label after `'` in leave")
                                .with_code_str(ErrCode::E004)
                                .with_span(self.span()));
                        }
                    }
                }
                _ => None,
            };
            self.expect(Token::Semicolon)?;
            let end = self.span().end;
            Ok(Stmt::Leave {
                label,
                span: Span::new(self.file_id, start, end),
            })
        }
    }

    // -------------------------------------------------------------------------
    // Continue
    // -------------------------------------------------------------------------

    pub(super) fn parse_continue_stmt(&mut self) -> Result<Stmt<'input>, Diagnostic<'src>> {
        let start = self.span().start;
        self.advance();
        let label = match self.peek().clone() {
            Some(Token::Apostrophe) => {
                self.advance();
                match self.advance() {
                    Some(Token::Ident(l)) => Some(l),
                    _ => {
                        return Err(Diagnostic::error("expected a label after `'` in continue")
                            .with_code_str(ErrCode::E004)
                            .with_span(self.span()));
                    }
                }
            }
            _ => None,
        };
        self.expect(Token::Semicolon)?;
        let end = self.span().end;
        Ok(Stmt::Continue {
            label,
            span: Span::new(self.file_id, start, end),
        })
    }

    // -------------------------------------------------------------------------
    // Return
    // -------------------------------------------------------------------------

    pub(super) fn parse_return_stmt(
        &mut self,
        bump: &'input bumpalo::Bump,
    ) -> Result<Stmt<'input>, Diagnostic<'src>> {
        let start = self.span().start;
        self.advance();
        let mut labels = Vec::new();
        while matches!(self.peek(), Some(Token::At)) {
            self.advance();
            match self.advance() {
                Some(Token::Ident(name)) => {
                    labels.push(Symbol::intern_prefixed('@', name.as_str()))
                }
                Some(tok) => {
                    return Err(Diagnostic::error(format!(
                        "expected label name after `@`, found {:?}",
                        tok
                    ))
                    .with_code_str(ErrCode::E004)
                    .with_help("a path label must be an identifier: `@label_name`")
                    .with_suggestion("write `return @even 4` instead of `return @ 4`")
                    .with_span(self.span()));
                }
                None => {
                    return Err(Diagnostic::error(
                        "unexpected end of file after `@` in return label",
                    )
                    .with_code_str(ErrCode::E002)
                    .with_span(self.span()));
                }
            }
        }
        let value = if !matches!(self.peek(), Some(Token::Semicolon) | Some(Token::RBrace)) {
            Some(self.parse_expr(bump)?)
        } else {
            None
        };
        self.expect(Token::Semicolon)?;
        let end = self.span().end;
        Ok(Stmt::Return {
            value,
            labels: bump.alloc_slice_clone(&labels),
            span: Span::new(self.file_id, start, end),
        })
    }

    // -------------------------------------------------------------------------
    // Scope cleanup
    // -------------------------------------------------------------------------

    pub(super) fn parse_scope_cleanup(
        &mut self,
        bump: &'input bumpalo::Bump,
    ) -> Result<Stmt<'input>, Diagnostic<'src>> {
        let start = self.span().start;
        self.advance();
        self.expect(Token::At)?;
        let name = match self.advance() {
            Some(Token::Ident(name)) => name,
            _ => return Err(Diagnostic::error("expected identifier for scope_cleanup")
                .with_code_str(ErrCode::E004)
                .with_help("`scope_cleanup` must be followed by `@<name>` — e.g. `scope_cleanup @cleanup { ... }`")
                .with_suggestion("use `scope_cleanup @identifier { body }` syntax")
                .with_span(self.span())),
        };
        let when_condition = if matches!(self.peek(), Some(Token::When)) {
            self.advance();
            let expr = self.with_restrictions(ParseRestrictions::NO_STRUCT_LITERAL, |this| {
                this.parse_expr(bump)
            })?;
            Some(Self::alloc_shared(bump, expr))
        } else {
            None
        };
        let mut propagates = false;
        let mut overrides = false;
        if matches!(self.peek(), Some(Token::Propagates)) {
            self.advance();
            propagates = true;
            if matches!(self.peek(), Some(Token::Overrides)) {
                self.advance();
                overrides = true;
            }
        } else if matches!(self.peek(), Some(Token::Overrides)) {
            return Err(
                Diagnostic::error("`overrides` must be used together with `propagates`")
                    .with_code_str(ErrCode::E004)
                    .with_suggestion("use both modifiers: `propagates overrides`")
                    .with_span(self.span()),
            );
        }
        self.expect(Token::LBrace)?;
        let body = self.parse_block(bump)?;
        self.expect(Token::RBrace)?;
        let end = self.span().end;
        Ok(Stmt::ScopeCleanup {
            name,
            when_condition,
            body,
            propagates,
            overrides,
            span: Span::new(self.file_id, start, end),
        })
    }

    // -------------------------------------------------------------------------
    // Trigger
    // -------------------------------------------------------------------------

    pub(super) fn parse_trigger(&mut self) -> Result<Stmt<'input>, Diagnostic<'src>> {
        let start = self.span().start;
        self.advance();
        self.expect(Token::At)?;
        let name = match self.advance() {
            Some(Token::Ident(name)) => name,
            _ => {
                return Err(Diagnostic::error("expected identifier for trigger")
                    .with_code_str(ErrCode::E004)
                    .with_help("`trigger` must be followed by `@<name>` — e.g. `trigger @cleanup;`")
                    .with_suggestion("use `trigger @identifier;` syntax")
                    .with_span(self.span()));
            }
        };
        self.expect(Token::Semicolon)?;
        let end = self.span().end;
        Ok(Stmt::Trigger {
            name,
            span: Span::new(self.file_id, start, end),
        })
    }

    // -------------------------------------------------------------------------
    // Unsafe block
    // -------------------------------------------------------------------------

    pub(super) fn parse_unsafe_block(
        &mut self,
        bump: &'input bumpalo::Bump,
    ) -> Result<Stmt<'input>, Diagnostic<'src>> {
        let start = self.span().start;
        self.advance();
        self.expect(Token::LBrace)?;
        let body = self.parse_block(bump)?;
        self.expect(Token::RBrace)?;
        let end = self.span().end;
        Ok(Stmt::Unsafe {
            body,
            span: Span::new(self.file_id, start, end),
        })
    }

    // -------------------------------------------------------------------------
    // Ghost variable
    // -------------------------------------------------------------------------

    pub(super) fn parse_ghost_variable(
        &mut self,
        bump: &'input bumpalo::Bump,
    ) -> Result<Stmt<'input>, Diagnostic<'src>> {
        let start = self.span().start;
        self.advance();
        let mut stmt = self.parse_variable_def(Vec::new(), bump)?;
        if let Stmt::VariableDef { .. } = &mut stmt {
            let end = self.span().end;
            return Ok(Stmt::GhostVariableDef {
                inner: Self::alloc_shared(bump, stmt),
                span: Span::new(self.file_id, start, end),
            });
        }
        Err(
            Diagnostic::error("expected variable definition after ghost")
                .with_code_str(ErrCode::E004)
                .with_help(
                    "`ghost` must be followed by a variable definition — e.g. `ghost set x = 0;`",
                )
                .with_suggestion("add a variable definition: `ghost set <name> = <value>;`")
                .with_span(self.span()),
        )
    }

    // -------------------------------------------------------------------------
    // Isolate block
    // -------------------------------------------------------------------------

    pub(super) fn parse_isolate_block(
        &mut self,
        bump: &'input bumpalo::Bump,
    ) -> Result<Stmt<'input>, Diagnostic<'src>> {
        let start = self.span().start;
        self.advance();
        self.expect(Token::LBrace)?;
        let body = self.parse_block(bump)?;
        self.expect(Token::RBrace)?;
        let end = self.span().end;
        Ok(Stmt::Isolate {
            attributes: &[],
            body,
            span: Span::new(self.file_id, start, end),
        })
    }
}
