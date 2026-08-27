//! Parsing of statements: blocks, variable definitions, control flow, scope_cleanup, etc.

use super::*;
use crate::ast::*;
use crate::diagnostics::Diagnostic;
use crate::lexer::Token;
use crate::symbol::Symbol;

impl<'input> Parser<'input> {
    // -------------------------------------------------------------------------
    // Block parsing
    // -------------------------------------------------------------------------

    pub(super) fn parse_block(&mut self) -> Result<Vec<Stmt<'input>>, Diagnostic> {
        self.recursion_depth += 1;
        if self.recursion_depth > self.max_recursion_depth {
            self.recursion_depth -= 1;
            return Err(Diagnostic::error(format!("maximum recursion depth {} exceeded", self.max_recursion_depth))
                .with_code_str("E006")
                .with_help("the parser reached its recursion limit — the type/structure may be deeply nested or self-referential")
                .with_suggestion("try breaking up deeply nested structures, or use less complex type expressions")
                .with_span(self.span()));
        }
        let result = self.parse_block_inner();
        self.recursion_depth -= 1;
        result
    }

    fn parse_block_inner(&mut self) -> Result<Vec<Stmt<'input>>, Diagnostic> {
        self.without_restrictions(ParseRestrictions::NO_STRUCT_LITERAL, |this| {
            let mut stmts = Vec::new();
            loop {
                match this.peek() {
                    Ok(Token::RBrace) | Err(()) => break,
                    _ => match this.parse_stmt() {
                        Ok(stmt) => stmts.push(stmt),
                        Err(diag) => {
                            this.diagnostics.push(diag);
                            this.synchronize();
                            stmts
                                .push(Stmt::Error(Span::new(this.span().start, this.span().start)));
                        }
                    },
                }
            }
            Ok(stmts)
        })
    }

    // -------------------------------------------------------------------------
    // Statement dispatch
    // -------------------------------------------------------------------------

    pub(super) fn parse_stmt(&mut self) -> Result<Stmt<'input>, Diagnostic> {
        self.recursion_depth += 1;
        if self.recursion_depth > self.max_recursion_depth {
            self.recursion_depth -= 1;
            return Err(Diagnostic::error(format!("maximum recursion depth {} exceeded", self.max_recursion_depth))
                .with_code_str("E006")
                .with_help("the parser reached its recursion limit — the type/structure may be deeply nested or self-referential")
                .with_suggestion("try breaking up deeply nested structures, or use less complex type expressions")
                .with_span(self.span()));
        }
        let result = self.parse_stmt_inner();
        self.recursion_depth -= 1;
        result
    }

    fn parse_stmt_inner(&mut self) -> Result<Stmt<'input>, Diagnostic> {
        // Collect leading @attributes — only valid before comptime inside
        // function bodies (e.g. @deprecated("use bar") comptime { ... }).
        // Use checkpoint/restore to avoid consuming `@` in expression-level
        // constructs like `@compile_error("msg")`.
        let mut attributes = Vec::new();
        if matches!(self.peek(), Ok(Token::At)) {
            let cp = self.checkpoint();
            loop {
                match self.peek() {
                    Ok(Token::At) => {
                        attributes.push(self.parse_attribute()?);
                    }
                    _ => break,
                }
            }
            if !matches!(self.peek(), Ok(Token::Comptime)) {
                // Not followed by comptime — restore cursor so the `@`
                // can be parsed as part of an expression (e.g. @compile_error).
                self.restore(&cp);
                attributes = Vec::new();
            }
        }
        // A loop label prefix: `'label: while ...` / `'label: for ...` /
        // `'label: loop ...`.  `continue 'label;` and `leave 'label;`
        // (SYNTAX.md §Loops) target the enclosing loop with that label.
        // Use checkpoint/restore so a stray apostrophe (e.g. in a lifetime
        // position) falls through to the normal statement dispatch.
        let label = if matches!(self.peek(), Ok(Token::Apostrophe)) {
            let cp = self.checkpoint();
            self.advance().ok();
            match self.advance() {
                Ok(Token::Ident(name)) if matches!(self.peek(), Ok(Token::Colon)) => {
                    self.advance().ok();
                    match self.peek() {
                        Ok(Token::While) | Ok(Token::For) | Ok(Token::Loop) => Some(name),
                        _ => {
                            self.restore(&cp);
                            None
                        }
                    }
                }
                _ => {
                    self.restore(&cp);
                    None
                }
            }
        } else {
            None
        };
        match self.peek() {
            Ok(Token::Set) | Ok(Token::Let) => self.parse_variable_def(attributes),
            Ok(Token::If) => self.parse_if_stmt(),
            Ok(Token::While) => self.parse_while_stmt(label),
            Ok(Token::For) => self.parse_for_stmt(label),
            Ok(Token::Loop) => self.parse_loop_stmt(label),
            Ok(Token::Leave) => self.parse_leave_stmt(),
            Ok(Token::Continue) => self.parse_continue_stmt(),
            Ok(Token::Return) => self.parse_return_stmt(),
            Ok(Token::LBrace) => {
                let start = self.span().start;
                self.advance().ok();
                let body = self.parse_block()?;
                self.expect(Token::RBrace)?;
                let end = self.span().end;
                Ok(Stmt::Expression(Expr::Block(body, Span::new(start, end))))
            }
            Ok(Token::Comptime) => self.parse_comptime_block_stmt(attributes),
            Ok(Token::Generate) => self.parse_generate_stmt(),
            Ok(Token::ScopeCleanup) => self.parse_scope_cleanup(),
            Ok(Token::Trigger) => self.parse_trigger(),
            Ok(Token::Unsafe) => self.parse_unsafe_block(),
            Ok(Token::Ghost) => self.parse_ghost_variable(),
            Ok(Token::Isolate) => self.parse_isolate_block(),
            Ok(Token::Match) => {
                let start = self.span().start;
                let expr = self.parse_match_expr()?;
                self.expect(Token::Semicolon)?;
                let end = self.span().end;
                Ok(Stmt::Expression(expr))
            }
            Ok(Token::Def) => {
                // Nested function definitions: parse them as function
                // definitions (rustc-style — items in blocks are collected
                // and referenced, not rejected).
                self.advance().ok();
                self.with_restrictions(ParseRestrictions::ALLOW_TYPE_PARAMS, |this| {
                    this.parse_function_def(Vec::new(), None, false, false)
                })
            }
            Ok(Token::Trait) => self.parse_trait_def(attributes, None),
            Ok(Token::Impl) => self
                .with_restrictions(ParseRestrictions::ALLOW_TYPE_PARAMS, |this| {
                    this.parse_impl_block(attributes)
                }),
            Ok(Token::Constraint) => self.parse_constraint(),
            Ok(Token::Type) => {
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
                let stmt = self
                    .with_restrictions(ParseRestrictions::ALLOW_TYPE_PARAMS, |this| {
                        this.parse_type_def(Vec::new(), None)
                    })?;
                // The Alias / Opaque branches of parse_type_def consume an
                // optional trailing `;`, but the Struct / Enum branches do
                // not.  In statement position users habitually terminate
                // with `;` — swallow it here so the leftover semicolon
                // does not trigger a spurious error.
                if matches!(self.peek(), Ok(Token::Semicolon)) {
                    self.advance().ok();
                }
                Ok(stmt)
            }
            _ => {
                let start = self.span().start;
                let lhs = self.parse_expr()?;
                if matches!(
                    self.peek(),
                    Ok(Token::Assign)
                        | Ok(Token::PlusEq)
                        | Ok(Token::MinusEq)
                        | Ok(Token::StarEq)
                        | Ok(Token::SlashEq)
                        | Ok(Token::ShlEq)
                        | Ok(Token::ShrEq)
                ) {
                    let op_token = self.advance().map_err(|_| Diagnostic::error("unexpected token")
                        .with_code_str("E003")
                        .with_help("expected an assignment operator (`=`, `+=`, `-=`, `*=`, `/=`, `<<=`, `>>=`) after the target")
                        .with_span(Span::new(0, 0)))?;
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
                    let value = self.parse_expr()?;
                    self.expect(Token::Semicolon)?;
                    let end = self.span().end;
                    Ok(Stmt::Assign {
                        target: Self::alloc_shared(self.arena, lhs),
                        op,
                        value,
                        span: Span::new(start, end),
                    })
                } else {
                    let at_end = matches!(self.peek(), Ok(Token::RBrace) | Err(()));
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
    ) -> Result<Stmt<'input>, Diagnostic> {
        let start = self.span().start;
        let kind = match self.advance().map_err(|_| {
            Diagnostic::error("unexpected token")
                .with_code_str("E003")
                .with_help("expected `set` or `let` to begin a variable definition")
                .with_span(Span::new(0, 0))
        })? {
            Token::Set => VariableKind::Set,
            Token::Let => VariableKind::Let,
            _ => unreachable!(),
        };
        let mutable = if kind == VariableKind::Set && matches!(self.peek(), Ok(Token::Mut)) {
            self.advance().ok();
            true
        } else {
            false
        };
        let (name, pattern) = if kind == VariableKind::Let
            && matches!(
                self.peek(),
                Ok(Token::LParen) | Ok(Token::LBracket) | Ok(Token::Ident(_))
            ) {
            if let Ok(Token::Ident(s)) = self.peek().clone() {
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
                    let ident = s;
                    self.advance().ok();
                    (Some(ident), None)
                } else {
                    (None, Some(self.parse_pattern()?))
                }
            } else {
                (None, Some(self.parse_pattern()?))
            }
        } else {
            let ident = match self.advance() {
                Ok(Token::Ident(name)) => name,
                Ok(Token::Auto) => Symbol::intern("auto"),
                Ok(tok) => return Err(Diagnostic::error(format!("expected variable name, found {:?}", tok))
                    .with_code_str("E004")
                    .with_help("a variable name must follow `set` or `let` — use a valid identifier")
                    .with_suggestion("use a valid identifier like `x` or `counter` for the variable name")
                    .with_span(self.span())),
                Err(()) => return Err(Diagnostic::error("unexpected end of file in variable definition")
                    .with_code_str("E002")
                    .with_help("variable definition is incomplete — expected a name or pattern after `set`/`let`")
                    .with_suggestion("add a variable name after `set`/`let`, e.g. `set x = 42;`")
                    .with_span(self.span())),
            };
            (Some(ident), None)
        };
        let ty = if matches!(self.peek(), Ok(Token::Colon)) {
            self.advance().ok();
            Some(self.parse_type()?)
        } else {
            None
        };
        let type_modifiers = if matches!(self.peek(), Ok(Token::With)) {
            self.parse_type_modifiers()?
        } else {
            Vec::new()
        };
        let type_captures = if let Some(ref name_str) = name {
            if name_str.eq_str("auto") && matches!(self.peek(), Ok(Token::Lt)) {
                self.parse_type_capture_params()?
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };
        let value = if matches!(self.peek(), Ok(Token::Assign)) {
            self.advance().ok();
            Some(self.parse_expr()?)
        } else {
            None
        };
        let else_branch = if kind == VariableKind::Let && matches!(self.peek(), Ok(Token::Else)) {
            self.advance().ok();
            self.expect(Token::LBrace)?;
            let block = self.parse_block()?;
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
            span: Span::new(start, end),
            attributes,
            doc: None,
            type_captures,
            type_modifiers,
        })
    }

    // -------------------------------------------------------------------------
    // `comptime` block as statement
    // -------------------------------------------------------------------------

    fn parse_comptime_block_stmt(
        &mut self,
        attributes: Vec<Attribute<'input>>,
    ) -> Result<Stmt<'input>, Diagnostic> {
        let start = self.span().start;
        self.advance().ok();
        let trusted = if matches!(self.peek(), Ok(Token::At)) {
            let attr = self.parse_attribute()?;
            if !attr.name.eq_str("trusted") {
                return Err(
                    Diagnostic::error("only `@trusted` is allowed after `comptime`")
                        .with_code_str("E004")
                        .with_span(attr.span)
                        .with_help("use `comptime @trusted { ... }` for a trusted comptime block"),
                );
            }
            true
        } else {
            false
        };
        let captures = if matches!(self.peek(), Ok(Token::LBracket)) {
            let mut captures = Vec::new();
            self.advance().ok();
            loop {
                let name_span = self.span();
                match self.peek() {
                    Ok(Token::RBracket) => {
                        self.advance().ok();
                        break;
                    }
                    Ok(Token::Ident(name)) => {
                        captures.push((*name, name_span));
                        self.advance().ok();
                        match self.peek() {
                            Ok(Token::Comma) => {
                                self.advance().ok();
                            }
                            Ok(Token::RBracket) => continue,
                            other => {
                                return Err(Diagnostic::error(format!(
                                    "expected ',' or ']' after capture name, found {:?}",
                                    other
                                ))
                                .with_code_str("E005")
                                .with_span(self.span()));
                            }
                        }
                    }
                    other => {
                        return Err(Diagnostic::error(format!(
                            "expected capture name or ']', found {:?}",
                            other
                        ))
                        .with_code_str("E005")
                        .with_span(self.span()));
                    }
                }
            }
            captures
        } else {
            Vec::new()
        };
        let trusted = if matches!(self.peek(), Ok(Token::At)) {
            let attr = self.parse_attribute()?;
            if !attr.name.eq_str("trusted") {
                return Err(Diagnostic::error("only `@trusted` is allowed after `comptime [captures]`")
                    .with_code_str("E004")
                    .with_span(attr.span)
                    .with_help("use `comptime [captures] @trusted { ... }` for a trusted comptime block with captures")
                    .with_suggestion("remove the unknown attribute, or use `@trusted`"));
            }
            true
        } else {
            trusted
        };
        self.expect(Token::LBrace)?;
        let body = self.parse_block()?;
        self.expect(Token::RBrace)?;
        let end = self.span().end;
        Ok(Stmt::ComptimeBlock {
            captures,
            trusted,
            attributes,
            body,
            span: Span::new(start, end),
        })
    }

    // -------------------------------------------------------------------------
    // `generate` block as statement
    // -------------------------------------------------------------------------

    fn parse_generate_stmt(&mut self) -> Result<Stmt<'input>, Diagnostic> {
        let start = self.span().start;
        self.advance().ok();
        match self.advance() {
            Ok(Token::For) => {}
            Ok(tok) => {
                return Err(Diagnostic::error(format!(
                    "expected `for` after `generate`, found {:?}",
                    tok
                ))
                .with_span(self.span()));
            }
            Err(()) => {
                return Err(
                    Diagnostic::error("expected `for` after `generate`").with_span(self.span())
                );
            }
        }
        let for_type = Self::alloc_shared(self.arena, self.parse_type()?);
        self.expect(Token::LBrace)?;
        let body = self.parse_block()?;
        self.expect(Token::RBrace)?;
        let end = self.span().end;
        Ok(Stmt::Generate {
            attributes: Vec::new(),
            for_type,
            body,
            span: Span::new(start, end),
        })
    }

    // -------------------------------------------------------------------------
    // If / if‑let
    // -------------------------------------------------------------------------

    pub(super) fn parse_if_stmt(&mut self) -> Result<Stmt<'input>, Diagnostic> {
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
            let then_branch = self.parse_block()?;
            self.expect(Token::RBrace)?;
            let else_branch = if matches!(self.peek(), Ok(Token::Else)) {
                self.advance().ok();
                if matches!(self.peek(), Ok(Token::If)) {
                    Some(vec![self.parse_if_stmt()?])
                } else {
                    self.expect(Token::LBrace)?;
                    let block = self.parse_block()?;
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
                span: Span::new(start, end),
            });
        }
        let cond = self.with_restrictions(ParseRestrictions::NO_STRUCT_LITERAL, |this| {
            this.parse_expr()
        })?;
        self.expect(Token::LBrace)?;
        let then_branch = self.parse_block()?;
        self.expect(Token::RBrace)?;
        let else_branch = if matches!(self.peek(), Ok(Token::Else)) {
            self.advance().ok();
            if matches!(self.peek(), Ok(Token::If)) {
                Some(vec![self.parse_if_stmt()?])
            } else {
                self.expect(Token::LBrace)?;
                let block = self.parse_block()?;
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
            span: Span::new(start, end),
        })
    }

    // -------------------------------------------------------------------------
    // While / while‑let
    // -------------------------------------------------------------------------

    pub(super) fn parse_while_stmt(
        &mut self,
        label: Option<Symbol>,
    ) -> Result<Stmt<'input>, Diagnostic> {
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
            let mut invariant = None;
            let mut decreases = None;
            while matches!(self.peek(), Ok(Token::Invariant) | Ok(Token::Decreases)) {
                match self.peek() {
                    Ok(Token::Invariant) => {
                        self.advance().ok();
                        let inv = self
                            .with_restrictions(ParseRestrictions::NO_STRUCT_LITERAL, |this| {
                                this.parse_expr()
                            })?;
                        invariant = Some(inv);
                    }
                    Ok(Token::Decreases) => {
                        self.advance().ok();
                        let dec = self
                            .with_restrictions(ParseRestrictions::NO_STRUCT_LITERAL, |this| {
                                this.parse_expr()
                            })?;
                        decreases = Some(dec);
                    }
                    _ => break,
                }
            }
            self.expect(Token::LBrace)?;
            let body = self.parse_block()?;
            self.expect(Token::RBrace)?;
            let end = self.span().end;
            return Ok(Stmt::WhileLet {
                label,
                pattern,
                scrutinee,
                body,
                invariant,
                decreases,
                span: Span::new(start, end),
            });
        }
        let cond = self.with_restrictions(ParseRestrictions::NO_STRUCT_LITERAL, |this| {
            this.parse_expr()
        })?;
        let mut invariant = None;
        let mut decreases = None;
        while matches!(self.peek(), Ok(Token::Invariant) | Ok(Token::Decreases)) {
            match self.peek() {
                Ok(Token::Invariant) => {
                    self.advance().ok();
                    let inv = self
                        .with_restrictions(ParseRestrictions::NO_STRUCT_LITERAL, |this| {
                            this.parse_expr()
                        })?;
                    invariant = Some(inv);
                }
                Ok(Token::Decreases) => {
                    self.advance().ok();
                    let dec = self
                        .with_restrictions(ParseRestrictions::NO_STRUCT_LITERAL, |this| {
                            this.parse_expr()
                        })?;
                    decreases = Some(dec);
                }
                _ => break,
            }
        }
        self.expect(Token::LBrace)?;
        let body = self.parse_block()?;
        self.expect(Token::RBrace)?;
        let end = self.span().end;
        Ok(Stmt::While {
            label,
            cond,
            body,
            invariant,
            decreases,
            span: Span::new(start, end),
        })
    }

    // -------------------------------------------------------------------------
    // For loop
    // -------------------------------------------------------------------------

    pub(super) fn parse_for_stmt(
        &mut self,
        label: Option<Symbol>,
    ) -> Result<Stmt<'input>, Diagnostic> {
        let start = self.span().start;
        self.advance().ok();
        let pattern = self.parse_pattern()?;
        self.expect(Token::In)?;
        let iterable = self.with_restrictions(ParseRestrictions::NO_STRUCT_LITERAL, |this| {
            this.parse_expr()
        })?;
        let mut invariant = None;
        let mut decreases = None;
        while matches!(self.peek(), Ok(Token::Invariant) | Ok(Token::Decreases)) {
            match self.peek() {
                Ok(Token::Invariant) => {
                    self.advance().ok();
                    let inv = self
                        .with_restrictions(ParseRestrictions::NO_STRUCT_LITERAL, |this| {
                            this.parse_expr()
                        })?;
                    invariant = Some(inv);
                }
                Ok(Token::Decreases) => {
                    self.advance().ok();
                    let dec = self
                        .with_restrictions(ParseRestrictions::NO_STRUCT_LITERAL, |this| {
                            this.parse_expr()
                        })?;
                    decreases = Some(dec);
                }
                _ => break,
            }
        }
        self.expect(Token::LBrace)?;
        let body = self.parse_block()?;
        self.expect(Token::RBrace)?;
        let end = self.span().end;
        Ok(Stmt::For {
            label,
            pattern,
            iterable,
            body,
            invariant,
            decreases,
            span: Span::new(start, end),
        })
    }

    // -------------------------------------------------------------------------
    // Loop
    // -------------------------------------------------------------------------

    pub(super) fn parse_loop_stmt(
        &mut self,
        label: Option<Symbol>,
    ) -> Result<Stmt<'input>, Diagnostic> {
        let start = self.span().start;
        self.advance().ok();
        self.expect(Token::LBrace)?;
        let body = self.parse_block()?;
        self.expect(Token::RBrace)?;
        let end = self.span().end;
        Ok(Stmt::Loop {
            label,
            body,
            span: Span::new(start, end),
        })
    }

    // -------------------------------------------------------------------------
    // Leave
    // -------------------------------------------------------------------------

    pub(super) fn parse_leave_stmt(&mut self) -> Result<Stmt<'input>, Diagnostic> {
        let start = self.span().start;
        self.advance().ok();
        if matches!(self.peek(), Ok(Token::With)) {
            self.advance().ok();
            let expr = self.parse_expr()?;
            self.expect(Token::Semicolon)?;
            let end = self.span().end;
            Ok(Stmt::Expression(Expr::LeaveWith {
                expr: Self::alloc_shared(self.arena, expr),
                is_return: false,
                span: Span::new(start, end),
            }))
        } else {
            let label = match self.peek().clone() {
                Ok(Token::Apostrophe) => {
                    self.advance().ok();
                    match self.advance() {
                        Ok(Token::Ident(l)) => Some(l),
                        _ => {
                            return Err(Diagnostic::error("expected a label after `'` in leave")
                                .with_code_str("E004")
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
                span: Span::new(start, end),
            })
        }
    }

    // -------------------------------------------------------------------------
    // Continue
    // -------------------------------------------------------------------------

    pub(super) fn parse_continue_stmt(&mut self) -> Result<Stmt<'input>, Diagnostic> {
        let start = self.span().start;
        self.advance().ok();
        let label = match self.peek().clone() {
            Ok(Token::Apostrophe) => {
                self.advance().ok();
                match self.advance() {
                    Ok(Token::Ident(l)) => Some(l),
                    _ => {
                        return Err(Diagnostic::error("expected a label after `'` in continue")
                            .with_code_str("E004")
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
            span: Span::new(start, end),
        })
    }

    // -------------------------------------------------------------------------
    // Return
    // -------------------------------------------------------------------------

    pub(super) fn parse_return_stmt(&mut self) -> Result<Stmt<'input>, Diagnostic> {
        let start = self.span().start;
        self.advance().ok();
        let mut labels = Vec::new();
        while matches!(self.peek(), Ok(Token::At)) {
            self.advance().ok();
            match self.advance() {
                Ok(Token::Ident(name)) => {
                    labels.push(Symbol::intern(&format!("@{}", name.as_str())))
                }
                Ok(tok) => {
                    return Err(Diagnostic::error(format!(
                        "expected label name after `@`, found {:?}",
                        tok
                    ))
                    .with_code_str("E004")
                    .with_help("a path label must be an identifier: `@label_name`")
                    .with_suggestion("write `return @even 4` instead of `return @ 4`")
                    .with_span(self.span()));
                }
                Err(()) => {
                    return Err(Diagnostic::error(
                        "unexpected end of file after `@` in return label",
                    )
                    .with_code_str("E002")
                    .with_span(self.span()));
                }
            }
        }
        let value = if !matches!(self.peek(), Ok(Token::Semicolon) | Ok(Token::RBrace)) {
            Some(self.parse_expr()?)
        } else {
            None
        };
        self.expect(Token::Semicolon)?;
        let end = self.span().end;
        Ok(Stmt::Return {
            value,
            labels,
            span: Span::new(start, end),
        })
    }

    // -------------------------------------------------------------------------
    // Scope cleanup
    // -------------------------------------------------------------------------

    pub(super) fn parse_scope_cleanup(&mut self) -> Result<Stmt<'input>, Diagnostic> {
        let start = self.span().start;
        self.advance().ok();
        self.expect(Token::At)?;
        let name = match self.advance() {
            Ok(Token::Ident(name)) => name,
            _ => return Err(Diagnostic::error("expected identifier for scope_cleanup")
                .with_code_str("E004")
                .with_help("`scope_cleanup` must be followed by `@<name>` — e.g. `scope_cleanup @cleanup { ... }`")
                .with_suggestion("use `scope_cleanup @identifier { body }` syntax")
                .with_span(self.span())),
        };
        let when_condition = if matches!(self.peek(), Ok(Token::When)) {
            self.advance().ok();
            let expr = self.with_restrictions(ParseRestrictions::NO_STRUCT_LITERAL, |this| {
                this.parse_expr()
            })?;
            Some(Self::alloc_shared(self.arena, expr))
        } else {
            None
        };
        let mut propagates = false;
        let mut overrides = false;
        if matches!(self.peek(), Ok(Token::Propagates)) {
            self.advance().ok();
            propagates = true;
            if matches!(self.peek(), Ok(Token::Overrides)) {
                self.advance().ok();
                overrides = true;
            }
        } else if matches!(self.peek(), Ok(Token::Overrides)) {
            return Err(
                Diagnostic::error("`overrides` must be used together with `propagates`")
                    .with_code_str("E004")
                    .with_suggestion("use both modifiers: `propagates overrides`")
                    .with_span(self.span()),
            );
        }
        self.expect(Token::LBrace)?;
        let body = self.parse_block()?;
        self.expect(Token::RBrace)?;
        let end = self.span().end;
        Ok(Stmt::ScopeCleanup {
            name,
            when_condition,
            body,
            propagates,
            overrides,
            span: Span::new(start, end),
        })
    }

    // -------------------------------------------------------------------------
    // Trigger
    // -------------------------------------------------------------------------

    pub(super) fn parse_trigger(&mut self) -> Result<Stmt<'input>, Diagnostic> {
        let start = self.span().start;
        self.advance().ok();
        self.expect(Token::At)?;
        let name = match self.advance() {
            Ok(Token::Ident(name)) => name,
            _ => {
                return Err(Diagnostic::error("expected identifier for trigger")
                    .with_code_str("E004")
                    .with_help("`trigger` must be followed by `@<name>` — e.g. `trigger @cleanup;`")
                    .with_suggestion("use `trigger @identifier;` syntax")
                    .with_span(self.span()));
            }
        };
        self.expect(Token::Semicolon)?;
        let end = self.span().end;
        Ok(Stmt::Trigger {
            name,
            span: Span::new(start, end),
        })
    }

    // -------------------------------------------------------------------------
    // Unsafe block
    // -------------------------------------------------------------------------

    pub(super) fn parse_unsafe_block(&mut self) -> Result<Stmt<'input>, Diagnostic> {
        let start = self.span().start;
        self.advance().ok();
        self.expect(Token::LBrace)?;
        let body = self.parse_block()?;
        self.expect(Token::RBrace)?;
        let end = self.span().end;
        Ok(Stmt::Unsafe {
            body,
            span: Span::new(start, end),
        })
    }

    // -------------------------------------------------------------------------
    // Ghost variable
    // -------------------------------------------------------------------------

    pub(super) fn parse_ghost_variable(&mut self) -> Result<Stmt<'input>, Diagnostic> {
        let start = self.span().start;
        self.advance().ok();
        let mut stmt = self.parse_variable_def(Vec::new())?;
        if let Stmt::VariableDef { .. } = &mut stmt {
            let end = self.span().end;
            return Ok(Stmt::GhostVariableDef {
                inner: Self::alloc_shared(self.arena, stmt),
                span: Span::new(start, end),
            });
        }
        Err(
            Diagnostic::error("expected variable definition after ghost")
                .with_code_str("E004")
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

    pub(super) fn parse_isolate_block(&mut self) -> Result<Stmt<'input>, Diagnostic> {
        let start = self.span().start;
        self.advance().ok();
        self.expect(Token::LBrace)?;
        let body = self.parse_block()?;
        self.expect(Token::RBrace)?;
        let end = self.span().end;
        Ok(Stmt::Isolate {
            attributes: Vec::new(),
            body,
            span: Span::new(start, end),
        })
    }
}
