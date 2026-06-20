use crate::ast::*;
use crate::lexer::Token;
use bitflags::bitflags;
use logos::Logos;

bitflags! {
    #[derive(Clone, Copy, Debug, Default)]
    pub struct ParseRestrictions: u8 {
        const NO_STRUCT_LITERAL = 1 << 0;
        const ALLOW_TYPE_PARAMS = 1 << 1;
        const STMT_EXPR         = 1 << 2;
        const VALUE_BLOCK       = 1 << 3;
    }
}

pub struct Parser<'source> {
    lexer: logos::Lexer<'source, Token>,
    peeked: Option<Result<Token, ()>>,
    pub diagnostics: Vec<Diagnostic>,
    recursion_depth: usize,
    max_recursion_depth: usize,
    restrictions: ParseRestrictions,
}

#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub message: String,
    pub span: Span,
}

impl<'source> Parser<'source> {
    pub fn new(source: &'source str) -> Self {
        Parser {
            lexer: Token::lexer(source),
            peeked: None,
            diagnostics: Vec::new(),
            recursion_depth: 0,
            max_recursion_depth: 256,
            restrictions: ParseRestrictions::STMT_EXPR,
        }
    }

    fn next_token(&mut self) -> Result<Token, ()> {
        loop {
            match self.lexer.next() {
                Some(Ok(Token::WhitespaceOrComment)) => continue,
                Some(Ok(other)) => return Ok(other),
                Some(Err(())) => return Err(()),
                None => return Err(()),
            }
        }
    }

    fn peek(&mut self) -> &Result<Token, ()> {
        if self.peeked.is_none() {
            self.peeked = Some(self.next_token());
        }
        self.peeked.as_ref().unwrap()
    }

    fn advance(&mut self) -> Result<Token, ()> {
        match self.peeked.take() {
            Some(tok) => tok,
            None => self.next_token(),
        }
    }

    fn span(&self) -> Span {
        let range = self.lexer.span();
        Span::new(range.start, range.end)
    }

    fn expect(&mut self, expected: Token) -> Result<Token, Diagnostic> {
        match self.advance() {
            Ok(tok) if tok == expected => Ok(tok),
            Ok(tok) => Err(Diagnostic {
                message: format!("expected {:?}, found {:?}", expected, tok),
                span: self.span(),
            }),
            Err(()) => Err(Diagnostic {
                message: "unexpected end of file".to_string(),
                span: self.span(),
            }),
        }
    }

    fn synchronize(&mut self) {
        loop {
            match self.peek() {
                Ok(Token::Semicolon) | Ok(Token::RBrace) => {
                    self.advance().ok();
                    return;
                }
                Ok(Token::Def) | Ok(Token::Set) | Ok(Token::Let) | Ok(Token::Type)
                | Ok(Token::Import) | Ok(Token::From) | Ok(Token::Extern) | Ok(Token::Edition)
                | Ok(Token::At) => return,
                Err(()) => return,
                _ => {
                    let tok = self.advance().ok();
                    self.diagnostics.push(Diagnostic {
                        message: format!("unexpected token during error recovery: {:?}", tok),
                        span: self.span(),
                    });
                }
            }
        }
    }

    fn with_restrictions<T>(
        &mut self,
        extra: ParseRestrictions,
        f: impl FnOnce(&mut Self) -> T,
    ) -> T {
        let old = self.restrictions;
        self.restrictions |= extra;
        let result = f(self);
        self.restrictions = old;
        result
    }

    fn without_restrictions<T>(
        &mut self,
        remove: ParseRestrictions,
        f: impl FnOnce(&mut Self) -> T,
    ) -> T {
        let old = self.restrictions;
        self.restrictions -= remove;
        let result = f(self);
        self.restrictions = old;
        result
    }

    pub fn parse_program(&mut self) -> Result<Program, Vec<Diagnostic>> {
        let start = self.span().start;
        let mut items = Vec::new();
        loop {
            match self.peek() {
                Err(()) => break,
                _ => match self.parse_item() {
                    Ok(item) => items.push(item),
                    Err(diag) => {
                        self.diagnostics.push(diag);
                        self.synchronize();
                    }
                },
            }
        }
        let end = self.span().end;
        let span = Span::new(start, end);
        if self.diagnostics.is_empty() {
            Ok(Program { items, span })
        } else {
            Err(std::mem::take(&mut self.diagnostics))
        }
    }

    fn parse_item(&mut self) -> Result<Stmt, Diagnostic> {
        let mut attributes = Vec::new();
        let mut doc = None;
        loop {
            match self.peek() {
                Ok(Token::At) => {
                    attributes.push(self.parse_attribute()?);
                }
                Ok(Token::DocComment(s)) => {
                    doc = Some(s.clone());
                    self.advance().ok();
                }
                _ => break,
            }
        }
        match self.peek() {
            Ok(Token::Def) => self
                .with_restrictions(ParseRestrictions::ALLOW_TYPE_PARAMS, |this| {
                    this.parse_function_def(attributes, doc)
                }),
            Ok(Token::Edition) => self.parse_edition(),
            Ok(Token::Import) | Ok(Token::From) => self.parse_import(),
            Ok(Token::Extern) => self.parse_extern_function(attributes),
            Ok(Token::Type) => self
                .with_restrictions(ParseRestrictions::ALLOW_TYPE_PARAMS, |this| {
                    this.parse_type_def(attributes, doc)
                }),
            Ok(Token::Impl) => self
                .with_restrictions(ParseRestrictions::ALLOW_TYPE_PARAMS, |this| {
                    this.parse_impl_block(attributes)
                }),
            _ => {
                let tok = self.advance().ok();
                Err(Diagnostic {
                    message: format!("unexpected token at top level: {:?}", tok),
                    span: self.span(),
                })
            }
        }
    }

    fn parse_attribute(&mut self) -> Result<Attribute, Diagnostic> {
        let start = self.span().start;
        self.advance().ok();
        let name = match self.advance() {
            Ok(Token::Ident(name)) => name,
            Ok(tok) => {
                return Err(Diagnostic {
                    message: format!("expected attribute name, found {:?}", tok),
                    span: self.span(),
                });
            }
            Err(()) => {
                return Err(Diagnostic {
                    message: "unexpected end of file in attribute".to_string(),
                    span: self.span(),
                });
            }
        };
        let args = if matches!(self.peek(), Ok(Token::LParen)) {
            self.advance().ok();
            let mut args = Vec::new();
            loop {
                if matches!(self.peek(), Ok(Token::RParen)) {
                    self.advance().ok();
                    break;
                }
                args.push(self.parse_expr()?);
                if matches!(self.peek(), Ok(Token::Comma)) {
                    self.advance().ok();
                } else {
                    self.expect(Token::RParen)?;
                    break;
                }
            }
            args
        } else {
            Vec::new()
        };
        let end = self.span().end;
        Ok(Attribute {
            name,
            args,
            span: Span::new(start, end),
        })
    }

    fn parse_function_def(
        &mut self,
        attributes: Vec<Attribute>,
        doc: Option<String>,
    ) -> Result<Stmt, Diagnostic> {
        let start = self.span().start;
        self.advance().ok();
        let name = match self.advance() {
            Ok(Token::Ident(name)) => name,
            Ok(tok) => {
                return Err(Diagnostic {
                    message: format!("expected function name, found {:?}", tok),
                    span: self.span(),
                });
            }
            Err(()) => {
                return Err(Diagnostic {
                    message: "unexpected end of file in function definition".to_string(),
                    span: self.span(),
                });
            }
        };
        let type_params = if self
            .restrictions
            .contains(ParseRestrictions::ALLOW_TYPE_PARAMS)
            && matches!(self.peek(), Ok(Token::Lt))
        {
            self.parse_type_params()?
        } else {
            Vec::new()
        };
        self.expect(Token::LParen)?;
        let mut params = Vec::new();
        loop {
            match self.peek() {
                Ok(Token::RParen) => {
                    self.advance().ok();
                    break;
                }
                _ => {
                    let param = self.parse_param()?;
                    params.push(param);
                    if matches!(self.peek(), Ok(Token::Comma)) {
                        self.advance().ok();
                    } else {
                        self.expect(Token::RParen)?;
                        break;
                    }
                }
            }
        }
        let return_type = if matches!(self.peek(), Ok(Token::Arrow)) {
            self.advance().ok();
            self.parse_type()?
        } else {
            Type::Never(self.span())
        };
        let mut contracts = Vec::new();
        while matches!(
            self.peek(),
            Ok(Token::Requires)
                | Ok(Token::Ensures)
                | Ok(Token::Invariant)
                | Ok(Token::Decreases)
                | Ok(Token::Terminates)
        ) {
            contracts.push(self.parse_contract()?);
        }
        if matches!(self.peek(), Ok(Token::Where)) {
            self.parse_where_clause()?;
        }
        self.expect(Token::LBrace)?;
        let body = self.parse_block()?;
        self.expect(Token::RBrace)?;
        let finally = if matches!(self.peek(), Ok(Token::Finally)) {
            self.advance().ok();
            self.expect(Token::LBrace)?;
            let block = self.parse_block()?;
            self.expect(Token::RBrace)?;
            Some(block)
        } else {
            None
        };
        let end = self.span().end;
        Ok(Stmt::FunctionDef {
            span: Span::new(start, end),
            attributes,
            contracts,
            doc,
            name,
            params,
            return_type,
            body: Some(body),
            type_params,
            where_clause: None,
            finally,
        })
    }

    fn parse_type_params(&mut self) -> Result<Vec<TypeParam>, Diagnostic> {
        self.advance().ok();
        let mut p = Vec::new();
        loop {
            let n = match self.advance() {
                Ok(Token::Ident(name)) => name,
                _ => {
                    return Err(Diagnostic {
                        message: "expected type parameter name".to_string(),
                        span: self.span(),
                    });
                }
            };
            let mut bounds = Vec::new();
            if matches!(self.peek(), Ok(Token::Colon)) {
                self.advance().ok();
                loop {
                    bounds.push(self.parse_type()?);
                    if !matches!(self.peek(), Ok(Token::Plus)) {
                        break;
                    }
                    self.advance().ok();
                }
            }
            p.push(TypeParam {
                name: n,
                bounds,
                span: Span::new(self.span().start, self.span().end),
            });
            match self.peek() {
                Ok(Token::Comma) => {
                    self.advance().ok();
                }
                Ok(Token::Gt) => {
                    self.advance().ok();
                    break;
                }
                _ => {
                    return Err(Diagnostic {
                        message: "expected ',' or '>'".to_string(),
                        span: self.span(),
                    });
                }
            }
        }
        Ok(p)
    }

    fn parse_where_clause(&mut self) -> Result<(), Diagnostic> {
        self.advance().ok();
        loop {
            self.parse_type()?;
            self.expect(Token::Colon)?;
            loop {
                self.parse_type()?;
                if !matches!(self.peek(), Ok(Token::Plus)) {
                    break;
                }
                self.advance().ok();
            }
            if !matches!(self.peek(), Ok(Token::Comma)) {
                break;
            }
            self.advance().ok();
        }
        Ok(())
    }

    fn parse_param(&mut self) -> Result<Param, Diagnostic> {
        let start = self.span().start;
        let name = match self.advance() {
            Ok(Token::Ident(name)) => name,
            Ok(tok) => {
                return Err(Diagnostic {
                    message: format!("expected parameter name, found {:?}", tok),
                    span: self.span(),
                });
            }
            Err(()) => {
                return Err(Diagnostic {
                    message: "unexpected end of file in parameter list".to_string(),
                    span: self.span(),
                });
            }
        };
        let ty = if matches!(self.peek(), Ok(Token::Colon)) {
            self.advance().ok();
            Some(self.parse_type()?)
        } else {
            None
        };
        let default = if matches!(self.peek(), Ok(Token::Assign)) {
            self.advance().ok();
            Some(self.parse_expr()?)
        } else {
            None
        };
        let end = self.span().end;
        Ok(Param {
            name,
            ty,
            default,
            span: Span::new(start, end),
        })
    }

    fn parse_contract(&mut self) -> Result<Contract, Diagnostic> {
        let start = self.span().start;
        match self.advance().unwrap() {
            Token::Requires => {
                let expr = self.parse_expr()?;
                let end = self.span().end;
                Ok(Contract::Requires(expr, Span::new(start, end)))
            }
            Token::Ensures => {
                let expr = self.parse_expr()?;
                let end = self.span().end;
                Ok(Contract::Ensures(expr, Span::new(start, end)))
            }
            Token::Invariant => {
                let expr = self.parse_expr()?;
                let end = self.span().end;
                Ok(Contract::Invariant(expr, Span::new(start, end)))
            }
            Token::Decreases => {
                let expr = self.parse_expr()?;
                let end = self.span().end;
                Ok(Contract::Decreases(expr, Span::new(start, end)))
            }
            Token::Terminates => {
                let expr = self.parse_expr()?;
                let end = self.span().end;
                Ok(Contract::Terminates(expr, Span::new(start, end)))
            }
            _ => unreachable!(),
        }
    }

    fn parse_type(&mut self) -> Result<Type, Diagnostic> {
        self.recursion_depth += 1;
        if self.recursion_depth > self.max_recursion_depth {
            self.recursion_depth -= 1;
            return Err(Diagnostic {
                message: format!(
                    "maximum recursion depth {} exceeded",
                    self.max_recursion_depth
                ),
                span: self.span(),
            });
        }
        let result = self.parse_type_inner();
        self.recursion_depth -= 1;
        result
    }

    fn parse_type_inner(&mut self) -> Result<Type, Diagnostic> {
        let start = self.span().start;
        match self.peek() {
            Ok(Token::Ampersand) => {
                self.advance().ok();
                let mutable = matches!(self.peek(), Ok(Token::Mut));
                if mutable {
                    self.advance().ok();
                }
                let ty = self.parse_type()?;
                let end = self.span().end;
                Ok(Type::Reference(
                    Box::new(ty),
                    mutable,
                    Span::new(start, end),
                ))
            }
            Ok(Token::Star) => {
                self.advance().ok();
                let ty = self.parse_type()?;
                let end = self.span().end;
                Ok(Type::Pointer(Box::new(ty), Span::new(start, end)))
            }
            Ok(Token::LBracket) => {
                self.advance().ok();
                let ty = self.parse_type()?;
                if matches!(self.peek(), Ok(Token::Semicolon)) {
                    self.advance().ok();
                    let size = self.parse_expr()?;
                    self.expect(Token::RBracket)?;
                    let end = self.span().end;
                    Ok(Type::Array(
                        Box::new(ty),
                        Box::new(size),
                        Span::new(start, end),
                    ))
                } else {
                    self.expect(Token::RBracket)?;
                    let end = self.span().end;
                    Ok(Type::Slice(Box::new(ty), Span::new(start, end)))
                }
            }
            Ok(Token::Dyn) => {
                self.advance().ok();
                let mut traits = Vec::new();
                loop {
                    let t = self.parse_type()?;
                    traits.push(t);
                    if !matches!(self.peek(), Ok(Token::Plus)) {
                        break;
                    }
                    self.advance().ok();
                }
                let end = self.span().end;
                Ok(Type::DynTrait(traits, Span::new(start, end)))
            }
            Ok(Token::Exists) => {
                self.advance().ok();
                let name = match self.advance() {
                    Ok(Token::Ident(n)) => n,
                    _ => {
                        return Err(Diagnostic {
                            message: "expected identifier after exists".to_string(),
                            span: self.span(),
                        });
                    }
                };
                self.expect(Token::Colon)?;
                let base = self.parse_type()?;
                self.expect(Token::Invariant)?;
                let invariant = Box::new(self.parse_expr()?);
                let end = self.span().end;
                Ok(Type::Exists {
                    name,
                    base: Box::new(base),
                    invariant,
                    span: Span::new(start, end),
                })
            }
            Ok(Token::IntLiteral(_)) | Ok(Token::HexLiteral(_)) | Ok(Token::BinLiteral(_)) => {
                let expr = self.parse_literal()?;
                let end = self.span().end;
                Ok(Type::Literal(Box::new(expr), Span::new(start, end)))
            }
            _ => match self.advance() {
                Ok(Token::Ident(name)) => {
                    let mut path = vec![name];
                    while matches!(self.peek(), Ok(Token::ColonColon)) {
                        self.advance().ok();
                        if let Ok(Token::Ident(part)) = self.advance() {
                            path.push(part);
                        } else {
                            return Err(Diagnostic {
                                message: "expected identifier after '::'".to_string(),
                                span: self.span(),
                            });
                        }
                    }
                    if matches!(self.peek(), Ok(Token::Lt)) {
                        self.advance().ok();
                        let mut args = Vec::new();
                        loop {
                            let arg = self.parse_type()?;
                            args.push(arg);
                            match self.peek() {
                                Ok(Token::Comma) => {
                                    self.advance().ok();
                                }
                                Ok(Token::Gt) => {
                                    self.advance().ok();
                                    break;
                                }
                                _ => {
                                    return Err(Diagnostic {
                                        message: "expected ',' or '>' in type parameters"
                                            .to_string(),
                                        span: self.span(),
                                    });
                                }
                            }
                        }
                        let end = self.span().end;
                        Ok(Type::Generic(
                            Box::new(Type::Path(path, Span::new(start, end))),
                            args,
                            Span::new(start, end),
                        ))
                    } else {
                        let end = self.span().end;
                        Ok(Type::Path(path, Span::new(start, end)))
                    }
                }
                Ok(Token::LParen) => {
                    if matches!(self.peek(), Ok(Token::RParen)) {
                        self.advance().ok();
                        let end = self.span().end;
                        Ok(Type::Tuple(Vec::new(), Span::new(start, end)))
                    } else {
                        let mut types = Vec::new();
                        loop {
                            let ty = self.parse_type()?;
                            types.push(ty);
                            match self.peek() {
                                Ok(Token::Comma) => {
                                    self.advance().ok();
                                }
                                Ok(Token::RParen) => {
                                    self.advance().ok();
                                    break;
                                }
                                _ => {
                                    return Err(Diagnostic {
                                        message: "expected ',' or ')' in tuple type".to_string(),
                                        span: self.span(),
                                    });
                                }
                            }
                        }
                        let end = self.span().end;
                        Ok(Type::Tuple(types, Span::new(start, end)))
                    }
                }
                Ok(Token::Bang) => {
                    let end = self.span().end;
                    Ok(Type::Never(Span::new(start, end)))
                }
                Ok(tok) => Err(Diagnostic {
                    message: format!("expected type, found {:?}", tok),
                    span: self.span(),
                }),
                Err(()) => Err(Diagnostic {
                    message: "unexpected end of file in type".to_string(),
                    span: self.span(),
                }),
            },
        }
    }

    fn parse_block(&mut self) -> Result<Vec<Stmt>, Diagnostic> {
        self.recursion_depth += 1;
        if self.recursion_depth > self.max_recursion_depth {
            self.recursion_depth -= 1;
            return Err(Diagnostic {
                message: format!(
                    "maximum recursion depth {} exceeded",
                    self.max_recursion_depth
                ),
                span: self.span(),
            });
        }
        let result = self.parse_block_inner();
        self.recursion_depth -= 1;
        result
    }

    fn parse_block_inner(&mut self) -> Result<Vec<Stmt>, Diagnostic> {
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

    fn parse_stmt(&mut self) -> Result<Stmt, Diagnostic> {
        self.recursion_depth += 1;
        if self.recursion_depth > self.max_recursion_depth {
            self.recursion_depth -= 1;
            return Err(Diagnostic {
                message: format!(
                    "maximum recursion depth {} exceeded",
                    self.max_recursion_depth
                ),
                span: self.span(),
            });
        }
        let result = self.parse_stmt_inner();
        self.recursion_depth -= 1;
        result
    }

    fn parse_stmt_inner(&mut self) -> Result<Stmt, Diagnostic> {
        match self.peek() {
            Ok(Token::Set) | Ok(Token::Let) => self.parse_variable_def(),
            Ok(Token::If) => self.parse_if_stmt(),
            Ok(Token::While) => self.parse_while_stmt(),
            Ok(Token::For) => self.parse_for_stmt(),
            Ok(Token::Loop) => self.parse_loop_stmt(),
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
            Ok(Token::Comptime) => self.parse_comptime_block(),
            Ok(Token::ScopeCleanup) => self.parse_scope_cleanup(),
            Ok(Token::Trigger) => self.parse_trigger(),
            Ok(Token::Unsafe) => self.parse_unsafe_block(),
            Ok(Token::Ghost) => self.parse_ghost_variable(),
            Ok(Token::Isolate) => self.parse_isolate_block(),
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
                ) {
                    let op_token = self.advance().unwrap();
                    let op = match op_token {
                        Token::Assign => None,
                        Token::PlusEq => Some(BinOp::Add),
                        Token::MinusEq => Some(BinOp::Sub),
                        Token::StarEq => Some(BinOp::Mul),
                        Token::SlashEq => Some(BinOp::Div),
                        _ => unreachable!(),
                    };
                    let value = self.parse_expr()?;
                    self.expect(Token::Semicolon)
                        .or_else(|_| Ok(Token::Semicolon))?;
                    let end = self.span().end;
                    Ok(Stmt::Assign {
                        target: Box::new(lhs),
                        op,
                        value,
                        span: Span::new(start, end),
                    })
                } else {
                    let at_end = matches!(self.peek(), Ok(Token::RBrace) | Err(()));
                    if at_end {
                        if self.restrictions.contains(ParseRestrictions::VALUE_BLOCK) {
                            Ok(Stmt::Expression(lhs))
                        } else {
                            self.expect(Token::Semicolon)
                                .or_else(|_| Ok(Token::Semicolon))?;
                            Ok(Stmt::Expression(lhs))
                        }
                    } else {
                        if self.restrictions.contains(ParseRestrictions::STMT_EXPR) {
                            self.expect(Token::Semicolon)
                                .or_else(|_| Ok(Token::Semicolon))?;
                        }
                        Ok(Stmt::Expression(lhs))
                    }
                }
            }
        }
    }

    fn parse_unsafe_block(&mut self) -> Result<Stmt, Diagnostic> {
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

    fn parse_ghost_variable(&mut self) -> Result<Stmt, Diagnostic> {
        let start = self.span().start;
        self.advance().ok();
        let mut stmt = self.parse_variable_def()?;
        if let Stmt::VariableDef { .. } = &mut stmt {
            let end = self.span().end;
            return Ok(Stmt::GhostVariableDef {
                inner: Box::new(stmt),
                span: Span::new(start, end),
            });
        }
        Err(Diagnostic {
            message: "expected variable definition after ghost".to_string(),
            span: self.span(),
        })
    }

    fn parse_isolate_block(&mut self) -> Result<Stmt, Diagnostic> {
        let start = self.span().start;
        self.advance().ok();
        self.expect(Token::LBrace)?;
        let body = self.parse_block()?;
        self.expect(Token::RBrace)?;
        let end = self.span().end;
        Ok(Stmt::Isolate {
            body,
            span: Span::new(start, end),
        })
    }

    fn parse_variable_def(&mut self) -> Result<Stmt, Diagnostic> {
        let start = self.span().start;
        let kind = match self.advance().unwrap() {
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
                    && (s == "_" || s.chars().next().map_or(false, |c| c.is_alphabetic()))
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
                Ok(tok) => {
                    return Err(Diagnostic {
                        message: format!("expected variable name, found {:?}", tok),
                        span: self.span(),
                    });
                }
                Err(()) => {
                    return Err(Diagnostic {
                        message: "unexpected end of file in variable definition".to_string(),
                        span: self.span(),
                    });
                }
            };
            (Some(ident), None)
        };
        let ty = if matches!(self.peek(), Ok(Token::Colon)) {
            self.advance().ok();
            Some(self.parse_type()?)
        } else {
            None
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
        self.expect(Token::Semicolon)
            .or_else(|_| Ok(Token::Semicolon))?;
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
            attributes: Vec::new(),
            doc: None,
        })
    }

    fn parse_if_stmt(&mut self) -> Result<Stmt, Diagnostic> {
        let start = self.span().start;
        self.advance().ok();
        if matches!(self.peek(), Ok(Token::Let)) {
            self.advance().ok();
            let pattern = self.parse_pattern()?;
            self.expect(Token::Assign)?;
            let old_restrict = self.restrictions;
            self.restrictions |= ParseRestrictions::NO_STRUCT_LITERAL;
            let scrutinee = self.parse_expr()?;
            self.restrictions = old_restrict;
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
        let old_restrict = self.restrictions;
        self.restrictions |= ParseRestrictions::NO_STRUCT_LITERAL;
        let cond = self.parse_expr()?;
        self.restrictions = old_restrict;
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

    fn parse_while_stmt(&mut self) -> Result<Stmt, Diagnostic> {
        let start = self.span().start;
        self.advance().ok();
        if matches!(self.peek(), Ok(Token::Let)) {
            self.advance().ok();
            let pattern = self.parse_pattern()?;
            self.expect(Token::Assign)?;
            let old_restrict = self.restrictions;
            self.restrictions |= ParseRestrictions::NO_STRUCT_LITERAL;
            let scrutinee = self.parse_expr()?;
            self.restrictions = old_restrict;
            self.expect(Token::LBrace)?;
            let body = self.parse_block()?;
            self.expect(Token::RBrace)?;
            let end = self.span().end;
            return Ok(Stmt::WhileLet {
                pattern,
                scrutinee,
                body,
                span: Span::new(start, end),
            });
        }
        let old_restrict = self.restrictions;
        self.restrictions |= ParseRestrictions::NO_STRUCT_LITERAL;
        let cond = self.parse_expr()?;
        self.restrictions = old_restrict;
        self.expect(Token::LBrace)?;
        let body = self.parse_block()?;
        self.expect(Token::RBrace)?;
        let end = self.span().end;
        Ok(Stmt::While {
            cond,
            body,
            span: Span::new(start, end),
        })
    }

    fn parse_for_stmt(&mut self) -> Result<Stmt, Diagnostic> {
        let start = self.span().start;
        self.advance().ok();
        let pattern = self.parse_pattern()?;
        self.expect(Token::In)?;
        let old_restrict = self.restrictions;
        self.restrictions |= ParseRestrictions::NO_STRUCT_LITERAL;
        let iterable = self.parse_expr()?;
        self.restrictions = old_restrict;
        self.expect(Token::LBrace)?;
        let body = self.parse_block()?;
        self.expect(Token::RBrace)?;
        let end = self.span().end;
        Ok(Stmt::For {
            pattern,
            iterable,
            body,
            span: Span::new(start, end),
        })
    }

    fn parse_loop_stmt(&mut self) -> Result<Stmt, Diagnostic> {
        let start = self.span().start;
        self.advance().ok();
        self.expect(Token::LBrace)?;
        let body = self.parse_block()?;
        self.expect(Token::RBrace)?;
        let end = self.span().end;
        Ok(Stmt::Loop {
            body,
            span: Span::new(start, end),
        })
    }

    fn parse_leave_stmt(&mut self) -> Result<Stmt, Diagnostic> {
        let start = self.span().start;
        self.advance().ok();
        let label = if let Ok(Token::Ident(l)) = self.peek().clone() {
            self.advance().ok();
            Some(l.clone())
        } else {
            None
        };
        self.expect(Token::Semicolon)
            .or_else(|_| Ok(Token::Semicolon))?;
        let end = self.span().end;
        Ok(Stmt::Leave {
            label,
            span: Span::new(start, end),
        })
    }

    fn parse_continue_stmt(&mut self) -> Result<Stmt, Diagnostic> {
        let start = self.span().start;
        self.advance().ok();
        let label = if let Ok(Token::Ident(l)) = self.peek().clone() {
            self.advance().ok();
            Some(l.clone())
        } else {
            None
        };
        self.expect(Token::Semicolon)
            .or_else(|_| Ok(Token::Semicolon))?;
        let end = self.span().end;
        Ok(Stmt::Continue {
            label,
            span: Span::new(start, end),
        })
    }

    fn parse_return_stmt(&mut self) -> Result<Stmt, Diagnostic> {
        let start = self.span().start;
        self.advance().ok();
        let value = if !matches!(self.peek(), Ok(Token::Semicolon) | Ok(Token::RBrace)) {
            Some(self.parse_expr()?)
        } else {
            None
        };
        self.expect(Token::Semicolon)
            .or_else(|_| Ok(Token::Semicolon))?;
        let end = self.span().end;
        Ok(Stmt::Return {
            value,
            span: Span::new(start, end),
        })
    }

    fn parse_comptime_block(&mut self) -> Result<Stmt, Diagnostic> {
        let start = self.span().start;
        self.advance().ok();
        self.expect(Token::LBrace)?;
        let body = self.parse_block()?;
        self.expect(Token::RBrace)?;
        let end = self.span().end;
        Ok(Stmt::ComptimeBlock {
            body,
            span: Span::new(start, end),
        })
    }

    fn parse_scope_cleanup(&mut self) -> Result<Stmt, Diagnostic> {
        let start = self.span().start;
        self.advance().ok();
        self.expect(Token::At)?;
        let name = match self.advance() {
            Ok(Token::Ident(name)) => name,
            _ => {
                return Err(Diagnostic {
                    message: "expected identifier for scope_cleanup".to_string(),
                    span: self.span(),
                });
            }
        };
        self.expect(Token::LBrace)?;
        let body = self.parse_block()?;
        self.expect(Token::RBrace)?;
        let end = self.span().end;
        Ok(Stmt::ScopeCleanup {
            name,
            body,
            span: Span::new(start, end),
        })
    }

    fn parse_trigger(&mut self) -> Result<Stmt, Diagnostic> {
        let start = self.span().start;
        self.advance().ok();
        self.expect(Token::At)?;
        let name = match self.advance() {
            Ok(Token::Ident(name)) => name,
            _ => {
                return Err(Diagnostic {
                    message: "expected identifier for trigger".to_string(),
                    span: self.span(),
                });
            }
        };
        self.expect(Token::Semicolon)
            .or_else(|_| Ok(Token::Semicolon))?;
        let end = self.span().end;
        Ok(Stmt::Trigger {
            name,
            span: Span::new(start, end),
        })
    }

    fn parse_edition(&mut self) -> Result<Stmt, Diagnostic> {
        let start = self.span().start;
        self.advance().ok();
        self.expect(Token::Assign)?;
        let edition = match self.advance() {
            Ok(Token::StringLiteral(Ok(s))) => s,
            Ok(tok) => {
                return Err(Diagnostic {
                    message: format!("expected edition string, found {:?}", tok),
                    span: self.span(),
                });
            }
            Err(()) => {
                return Err(Diagnostic {
                    message: "unexpected end of file in edition declaration".to_string(),
                    span: self.span(),
                });
            }
        };
        self.expect(Token::Semicolon)?;
        let end = self.span().end;
        Ok(Stmt::Edition(edition, Span::new(start, end)))
    }

    fn parse_import(&mut self) -> Result<Stmt, Diagnostic> {
        let start = self.span().start;
        let is_from = matches!(self.peek(), Ok(Token::From));
        if is_from {
            self.advance().ok();
        }
        let mut path = Vec::new();
        match self.advance() {
            Ok(Token::Ident(part)) => path.push(part),
            _ => {
                return Err(Diagnostic {
                    message: "expected module path".to_string(),
                    span: self.span(),
                });
            }
        }
        while matches!(self.peek(), Ok(Token::ColonColon)) {
            self.advance().ok();
            match self.advance() {
                Ok(Token::Ident(part)) => path.push(part),
                _ => {
                    return Err(Diagnostic {
                        message: "expected identifier after '::'".to_string(),
                        span: self.span(),
                    });
                }
            }
        }
        if matches!(self.peek(), Ok(Token::LBrace)) {
            self.advance().ok();
            let mut items = Vec::new();
            loop {
                if matches!(self.peek(), Ok(Token::RBrace)) {
                    self.advance().ok();
                    break;
                }
                items.push(match self.advance() {
                    Ok(Token::Ident(item)) => item,
                    _ => {
                        return Err(Diagnostic {
                            message: "expected import item name".to_string(),
                            span: self.span(),
                        });
                    }
                });
                if matches!(self.peek(), Ok(Token::Comma)) {
                    self.advance().ok();
                } else {
                    self.expect(Token::RBrace)?;
                    break;
                }
            }
            let alias = if matches!(self.peek(), Ok(Token::As)) {
                self.advance().ok();
                match self.advance() {
                    Ok(Token::Ident(a)) => Some(a),
                    _ => {
                        return Err(Diagnostic {
                            message: "expected alias name".to_string(),
                            span: self.span(),
                        });
                    }
                }
            } else {
                None
            };
            self.expect(Token::Semicolon)?;
            let end = self.span().end;
            return Ok(Stmt::Import {
                path,
                items: Some(items),
                alias,
                span: Span::new(start, end),
            });
        }
        let items = if is_from && matches!(self.peek(), Ok(Token::Import)) {
            self.advance().ok();
            self.expect(Token::LBrace)?;
            let mut items = Vec::new();
            loop {
                match self.advance() {
                    Ok(Token::Ident(item)) => items.push(item),
                    _ => {
                        return Err(Diagnostic {
                            message: "expected import item name".to_string(),
                            span: self.span(),
                        });
                    }
                }
                match self.peek() {
                    Ok(Token::Comma) => {
                        self.advance().ok();
                    }
                    Ok(Token::RBrace) => {
                        self.advance().ok();
                        break;
                    }
                    _ => {
                        return Err(Diagnostic {
                            message: "expected ',' or '}' in import list".to_string(),
                            span: self.span(),
                        });
                    }
                }
            }
            Some(items)
        } else {
            None
        };
        let alias = if matches!(self.peek(), Ok(Token::As)) {
            self.advance().ok();
            match self.advance() {
                Ok(Token::Ident(a)) => Some(a),
                _ => {
                    return Err(Diagnostic {
                        message: "expected alias name".to_string(),
                        span: self.span(),
                    });
                }
            }
        } else {
            None
        };
        self.expect(Token::Semicolon)?;
        let end = self.span().end;
        Ok(Stmt::Import {
            path,
            items,
            alias,
            span: Span::new(start, end),
        })
    }

    fn parse_extern_function(&mut self, attributes: Vec<Attribute>) -> Result<Stmt, Diagnostic> {
        let start = self.span().start;
        self.advance().ok();
        let abi = match self.advance() {
            Ok(Token::StringLiteral(Ok(s))) => s,
            _ => {
                return Err(Diagnostic {
                    message: "expected ABI string after 'extern'".to_string(),
                    span: self.span(),
                });
            }
        };
        self.expect(Token::Def)?;
        let name = match self.advance() {
            Ok(Token::Ident(name)) => name,
            _ => {
                return Err(Diagnostic {
                    message: "expected function name".to_string(),
                    span: self.span(),
                });
            }
        };
        self.expect(Token::LParen)?;
        let mut params = Vec::new();
        loop {
            match self.peek() {
                Ok(Token::RParen) => {
                    self.advance().ok();
                    break;
                }
                _ => {
                    let param = self.parse_param()?;
                    params.push(param);
                    if matches!(self.peek(), Ok(Token::Comma)) {
                        self.advance().ok();
                    } else {
                        self.expect(Token::RParen)?;
                        break;
                    }
                }
            }
        }
        let return_type = if matches!(self.peek(), Ok(Token::Arrow)) {
            self.advance().ok();
            self.parse_type()?
        } else {
            Type::Never(self.span())
        };
        self.expect(Token::Semicolon)?;
        let end = self.span().end;
        Ok(Stmt::ExternFunction {
            abi,
            name,
            params,
            return_type,
            span: Span::new(start, end),
            attributes,
        })
    }

    fn parse_type_def(
        &mut self,
        attributes: Vec<Attribute>,
        doc: Option<String>,
    ) -> Result<Stmt, Diagnostic> {
        let start = self.span().start;
        self.advance().ok();
        let name = match self.advance() {
            Ok(Token::Ident(name)) => name,
            _ => {
                return Err(Diagnostic {
                    message: "expected type name".to_string(),
                    span: self.span(),
                });
            }
        };
        let params = if self
            .restrictions
            .contains(ParseRestrictions::ALLOW_TYPE_PARAMS)
            && matches!(self.peek(), Ok(Token::Lt))
        {
            self.parse_type_params()?
        } else {
            Vec::new()
        };
        self.expect(Token::Assign)?;
        let definition = if let Ok(Token::Ident(s)) = self.peek().clone() {
            match s.as_str() {
                "struct" => {
                    self.advance().ok();
                    self.expect(Token::LBrace)?;
                    let mut fields = Vec::new();
                    loop {
                        if matches!(self.peek(), Ok(Token::RBrace)) {
                            self.advance().ok();
                            break;
                        }
                        let field_name = match self.advance() {
                            Ok(Token::Ident(n)) => n,
                            _ => {
                                return Err(Diagnostic {
                                    message: "expected field name".to_string(),
                                    span: self.span(),
                                });
                            }
                        };
                        self.expect(Token::Colon)?;
                        let field_ty = self.parse_type()?;
                        let default = if matches!(self.peek(), Ok(Token::Assign)) {
                            self.advance().ok();
                            Some(self.parse_expr()?)
                        } else {
                            None
                        };
                        fields.push(StructField {
                            name: field_name,
                            ty: field_ty,
                            default,
                            span: Span::new(start, self.span().end),
                        });
                        if matches!(self.peek(), Ok(Token::Comma)) {
                            self.advance().ok();
                        } else {
                            self.expect(Token::RBrace)?;
                            break;
                        }
                    }
                    TypeDefinition::Struct(fields)
                }
                "enum" => {
                    self.advance().ok();
                    self.expect(Token::LBrace)?;
                    let mut variants = Vec::new();
                    loop {
                        if matches!(self.peek(), Ok(Token::RBrace)) {
                            self.advance().ok();
                            break;
                        }
                        let v_name = match self.advance() {
                            Ok(Token::Ident(n)) => n,
                            _ => {
                                return Err(Diagnostic {
                                    message: "expected variant name".to_string(),
                                    span: self.span(),
                                });
                            }
                        };
                        let payload = if matches!(self.peek(), Ok(Token::LParen)) {
                            self.advance().ok();
                            let ty = self.parse_type()?;
                            self.expect(Token::RParen)?;
                            Some(ty)
                        } else {
                            None
                        };
                        variants.push(EnumVariant {
                            name: v_name,
                            payload,
                            span: Span::new(start, self.span().end),
                        });
                        if matches!(self.peek(), Ok(Token::Comma)) {
                            self.advance().ok();
                        } else {
                            self.expect(Token::RBrace)?;
                            break;
                        }
                    }
                    TypeDefinition::Enum(variants)
                }
                _ => {
                    let ty = self.parse_type()?;
                    if matches!(self.peek(), Ok(Token::Semicolon)) {
                        self.advance().ok();
                    } else if matches!(self.peek(), Ok(Token::With)) {
                        while matches!(self.peek(), Ok(Token::With)) {
                            self.advance().ok();
                            while !matches!(self.peek(), Ok(Token::Semicolon) | Err(())) {
                                self.advance().ok();
                            }
                            if matches!(self.peek(), Ok(Token::Semicolon)) {
                                self.advance().ok();
                            }
                        }
                    } else {
                        return Err(Diagnostic {
                            message: "expected ';' or 'with' after type alias".to_string(),
                            span: self.span(),
                        });
                    }
                    TypeDefinition::Alias(ty)
                }
            }
        } else {
            let ty = self.parse_type()?;
            if matches!(self.peek(), Ok(Token::Semicolon)) {
                self.advance().ok();
            } else if matches!(self.peek(), Ok(Token::With)) {
                while matches!(self.peek(), Ok(Token::With)) {
                    self.advance().ok();
                    while !matches!(self.peek(), Ok(Token::Semicolon) | Err(())) {
                        self.advance().ok();
                    }
                    if matches!(self.peek(), Ok(Token::Semicolon)) {
                        self.advance().ok();
                    }
                }
            } else {
                return Err(Diagnostic {
                    message: "expected ';' or 'with' after type alias".to_string(),
                    span: self.span(),
                });
            }
            TypeDefinition::Alias(ty)
        };
        let end = self.span().end;
        Ok(Stmt::TypeDef {
            span: Span::new(start, end),
            attributes,
            doc,
            name,
            params,
            definition,
            contracts: Vec::new(),
        })
    }

    fn parse_impl_block(&mut self, attributes: Vec<Attribute>) -> Result<Stmt, Diagnostic> {
        let start = self.span().start;
        self.advance().ok();
        let trait_path = if matches!(self.peek(), Ok(Token::Ident(_))) {
            let mut path = Vec::new();
            path.push(match self.advance() {
                Ok(Token::Ident(name)) => name,
                _ => {
                    return Err(Diagnostic {
                        message: "expected trait name".to_string(),
                        span: self.span(),
                    });
                }
            });
            while matches!(self.peek(), Ok(Token::ColonColon)) {
                self.advance().ok();
                path.push(match self.advance() {
                    Ok(Token::Ident(part)) => part,
                    _ => {
                        return Err(Diagnostic {
                            message: "expected identifier after '::'".to_string(),
                            span: self.span(),
                        });
                    }
                });
            }
            self.expect(Token::For)?;
            Some(path)
        } else {
            None
        };
        let for_type = self.parse_type()?;
        self.expect(Token::LBrace)?;
        let mut methods = Vec::new();
        loop {
            if matches!(self.peek(), Ok(Token::RBrace)) {
                break;
            }
            methods.push(self.parse_impl_method()?);
        }
        self.expect(Token::RBrace)?;
        let end = self.span().end;
        Ok(Stmt::ImplBlock {
            span: Span::new(start, end),
            attributes,
            trait_path,
            for_type,
            methods,
        })
    }

    fn parse_impl_method(&mut self) -> Result<ImplMethod, Diagnostic> {
        if matches!(self.peek(), Ok(Token::Def)) {
            self.advance().ok();
        }
        let start = self.span().start;
        let name = match self.advance() {
            Ok(Token::Ident(name)) => name,
            Ok(tok) => {
                return Err(Diagnostic {
                    message: format!("expected method name, found {:?}", tok),
                    span: self.span(),
                });
            }
            Err(()) => {
                return Err(Diagnostic {
                    message: "unexpected end of file in method definition".to_string(),
                    span: self.span(),
                });
            }
        };
        self.expect(Token::LParen)?;
        let mut params = Vec::new();
        loop {
            match self.peek() {
                Ok(Token::RParen) => {
                    self.advance().ok();
                    break;
                }
                Ok(Token::Ampersand) | Ok(Token::Ident(_)) => {
                    let param = self.parse_self_param()?;
                    params.push(param);
                }
                _ => {
                    let param = self.parse_param()?;
                    params.push(param);
                }
            }
            if matches!(self.peek(), Ok(Token::Comma)) {
                self.advance().ok();
            } else {
                self.expect(Token::RParen)?;
                break;
            }
        }
        let return_type = if matches!(self.peek(), Ok(Token::Arrow)) {
            self.advance().ok();
            self.parse_type()?
        } else {
            Type::Never(self.span())
        };
        let body = if matches!(self.peek(), Ok(Token::LBrace)) {
            self.advance().ok();
            let block = self.parse_block()?;
            self.expect(Token::RBrace)?;
            Some(block)
        } else {
            None
        };
        let end = self.span().end;
        Ok(ImplMethod {
            name,
            params,
            return_type,
            body,
            span: Span::new(start, end),
        })
    }

    fn parse_self_param(&mut self) -> Result<Param, Diagnostic> {
        let start = self.span().start;
        let mutable = if matches!(self.peek(), Ok(Token::Ampersand)) {
            self.advance().ok();
            let m = matches!(self.peek(), Ok(Token::Mut));
            if m {
                self.advance().ok();
            }
            m
        } else {
            false
        };
        match self.advance() {
            Ok(Token::Ident(s)) if s == "self" => {
                let end = self.span().end;
                let ty = if mutable {
                    Type::Reference(
                        Box::new(Type::Path(vec!["Self".into()], Span::new(start, end))),
                        true,
                        Span::new(start, end),
                    )
                } else {
                    Type::Path(vec!["Self".into()], Span::new(start, end))
                };
                Ok(Param {
                    name: "self".into(),
                    ty: Some(ty),
                    default: None,
                    span: Span::new(start, end),
                })
            }
            _ => Err(Diagnostic {
                message: "expected 'self'".to_string(),
                span: self.span(),
            }),
        }
    }

    fn parse_pattern(&mut self) -> Result<Pattern, Diagnostic> {
        self.recursion_depth += 1;
        if self.recursion_depth > self.max_recursion_depth {
            self.recursion_depth -= 1;
            return Err(Diagnostic {
                message: format!(
                    "maximum recursion depth {} exceeded",
                    self.max_recursion_depth
                ),
                span: self.span(),
            });
        }
        let result = self.parse_pattern_inner();
        self.recursion_depth -= 1;
        result
    }

    fn parse_pattern_inner(&mut self) -> Result<Pattern, Diagnostic> {
        let start = self.span().start;
        let tok = match self.peek() {
            Ok(t) => t.clone(),
            Err(()) => {
                return Err(Diagnostic {
                    message: "unexpected end of file in pattern".into(),
                    span: self.span(),
                });
            }
        };
        match tok {
            Token::IntLiteral(_)
            | Token::FloatLiteral(_)
            | Token::StringLiteral(_)
            | Token::ByteStringLiteral(_)
            | Token::CharLiteral(_)
            | Token::True
            | Token::False => {
                let lit = self.parse_literal()?;
                Ok(Pattern::Literal(
                    Box::new(lit),
                    Span::new(start, self.span().end),
                ))
            }
            Token::Ident(s) if s == "_" => {
                self.advance().ok();
                Ok(Pattern::Wildcard(Span::new(start, self.span().end)))
            }
            Token::Ident(_) => {
                let name = match self.advance() {
                    Ok(Token::Ident(n)) => n,
                    _ => unreachable!(),
                };
                if matches!(self.peek(), Ok(Token::LBrace)) {
                    self.advance().ok();
                    let mut fields = Vec::new();
                    loop {
                        if matches!(self.peek(), Ok(Token::RBrace)) {
                            self.advance().ok();
                            break;
                        }
                        let field_name = match self.advance() {
                            Ok(Token::Ident(f)) => f,
                            _ => {
                                return Err(Diagnostic {
                                    message: "expected field name".to_string(),
                                    span: self.span(),
                                });
                            }
                        };
                        let field_pattern = if matches!(self.peek(), Ok(Token::Colon)) {
                            self.advance().ok();
                            self.parse_pattern()?
                        } else {
                            Pattern::Ident(field_name.clone(), self.span())
                        };
                        fields.push((field_name, field_pattern));
                        if matches!(self.peek(), Ok(Token::Comma)) {
                            self.advance().ok();
                        } else {
                            self.expect(Token::RBrace)?;
                            break;
                        }
                    }
                    let end = self.span().end;
                    Ok(Pattern::Struct {
                        path: vec![name],
                        fields,
                        span: Span::new(start, end),
                    })
                } else if matches!(self.peek(), Ok(Token::LParen)) {
                    self.advance().ok();
                    let inner = self.parse_pattern()?;
                    self.expect(Token::RParen)?;
                    let end = self.span().end;
                    Ok(Pattern::Enum {
                        path: vec![],
                        variant: name,
                        inner: Some(Box::new(inner)),
                        span: Span::new(start, end),
                    })
                } else if matches!(self.peek(), Ok(Token::ColonColon)) {
                    let mut path = vec![name];
                    self.advance().ok();
                    path.push(match self.advance() {
                        Ok(Token::Ident(variant)) => variant,
                        _ => {
                            return Err(Diagnostic {
                                message: "expected variant name".to_string(),
                                span: self.span(),
                            });
                        }
                    });
                    let inner = if matches!(self.peek(), Ok(Token::LParen)) {
                        self.advance().ok();
                        let p = self.parse_pattern()?;
                        self.expect(Token::RParen)?;
                        Some(Box::new(p))
                    } else {
                        None
                    };
                    let variant = path.pop().unwrap();
                    let end = self.span().end;
                    Ok(Pattern::Enum {
                        path,
                        variant,
                        inner,
                        span: Span::new(start, end),
                    })
                } else {
                    Ok(Pattern::Ident(name, Span::new(start, self.span().end)))
                }
            }
            Token::LParen => {
                self.advance().ok();
                let mut patterns = Vec::new();
                loop {
                    if matches!(self.peek(), Ok(Token::RParen)) {
                        self.advance().ok();
                        break;
                    }
                    patterns.push(self.parse_pattern()?);
                    if matches!(self.peek(), Ok(Token::Comma)) {
                        self.advance().ok();
                    } else {
                        self.expect(Token::RParen)?;
                        break;
                    }
                }
                Ok(Pattern::Tuple(patterns, Span::new(start, self.span().end)))
            }
            _ => Err(Diagnostic {
                message: "expected pattern".to_string(),
                span: self.span(),
            }),
        }
    }

    fn parse_expr(&mut self) -> Result<Expr, Diagnostic> {
        self.parse_expr_bp(0)
    }

    fn parse_expr_bp(&mut self, min_bp: u8) -> Result<Expr, Diagnostic> {
        self.recursion_depth += 1;
        if self.recursion_depth > self.max_recursion_depth {
            self.recursion_depth -= 1;
            return Err(Diagnostic {
                message: format!(
                    "maximum recursion depth {} exceeded",
                    self.max_recursion_depth
                ),
                span: self.span(),
            });
        }
        let result = self.parse_expr_bp_inner(min_bp);
        self.recursion_depth -= 1;
        result
    }

    fn parse_expr_bp_inner(&mut self, min_bp: u8) -> Result<Expr, Diagnostic> {
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
                lhs = self.parse_infix(lhs, lbp)?;
            } else {
                break;
            }
        }
        Ok(lhs)
    }

    fn prefix_binding_power(&self, token: Option<&Token>) -> Option<(u8, bool)> {
        match token {
            Some(Token::Plus) | Some(Token::Minus) => Some((15, false)),
            Some(Token::Star) | Some(Token::Slash) | Some(Token::Percent) => Some((13, true)),
            Some(Token::PlusWrap) | Some(Token::MinusWrap) | Some(Token::StarWrap) => {
                Some((13, true))
            }
            Some(Token::PlusSaturate) | Some(Token::MinusSaturate) | Some(Token::StarSaturate) => {
                Some((13, true))
            }
            Some(Token::PlusTrap) | Some(Token::MinusTrap) | Some(Token::StarTrap) => {
                Some((13, true))
            }
            Some(Token::Ampersand) => Some((11, true)),
            Some(Token::Pipe) => Some((9, true)),
            Some(Token::Caret) => Some((10, true)),
            Some(Token::Shl) | Some(Token::Shr) => Some((12, true)),
            Some(Token::EqEq) | Some(Token::Neq) | Some(Token::Lt) | Some(Token::Gt)
            | Some(Token::Le) | Some(Token::Ge) => Some((8, true)),
            Some(Token::And) => Some((7, true)),
            Some(Token::Or) => Some((6, true)),
            Some(Token::DotDot) | Some(Token::DotDotEq) => Some((1, true)),
            Some(Token::LParen) => Some((18, true)),
            Some(Token::LBracket) => Some((18, true)),
            Some(Token::Dot) => Some((18, true)),
            Some(Token::Apostrophe) => Some((18, true)),
            Some(Token::Question) => Some((17, true)),
            Some(Token::Bang) => Some((16, false)),
            Some(Token::Not) => Some((16, false)),
            Some(Token::Tilde) => Some((16, false)),
            Some(Token::As) => Some((14, true)),
            _ => None,
        }
    }

    fn parse_prefix(&mut self) -> Result<Expr, Diagnostic> {
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
            | Ok(Token::False) => self.parse_literal(),
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
            Ok(Token::If) => self.parse_if_expr(),
            Ok(Token::Match) => self.parse_match_expr(),
            Ok(Token::Leave) => {
                self.advance().ok();
                self.expect(Token::With)?;
                let expr = self.parse_expr()?;
                let end = self.span().end;
                Ok(Expr::LeaveWith {
                    expr: Box::new(expr),
                    span: Span::new(start, end),
                })
            }
            Ok(Token::Await) => {
                self.advance().ok();
                let expr = self.parse_expr()?;
                let end = self.span().end;
                Ok(Expr::Await {
                    expr: Box::new(expr),
                    span: Span::new(start, end),
                })
            }
            Ok(Token::Bang) => {
                self.advance().ok();
                let expr = self.parse_prefix()?;
                let end = self.span().end;
                Ok(Expr::UnaryOp {
                    op: UnaryOp::Not,
                    expr: Box::new(expr),
                    span: Span::new(start, end),
                })
            }
            Ok(Token::Minus) => {
                self.advance().ok();
                let expr = self.parse_prefix()?;
                let end = self.span().end;
                Ok(Expr::UnaryOp {
                    op: UnaryOp::Neg,
                    expr: Box::new(expr),
                    span: Span::new(start, end),
                })
            }
            Ok(Token::Tilde) => {
                self.advance().ok();
                let expr = self.parse_prefix()?;
                let end = self.span().end;
                Ok(Expr::UnaryOp {
                    op: UnaryOp::BitNot,
                    expr: Box::new(expr),
                    span: Span::new(start, end),
                })
            }
            Ok(Token::Ampersand) => {
                self.advance().ok();
                let mutable = matches!(self.peek(), Ok(Token::Mut));
                if mutable {
                    self.advance().ok();
                }
                let expr = self.parse_prefix()?;
                let end = self.span().end;
                Ok(Expr::UnaryOp {
                    op: if mutable {
                        UnaryOp::RefMut
                    } else {
                        UnaryOp::Ref
                    },
                    expr: Box::new(expr),
                    span: Span::new(start, end),
                })
            }
            Ok(Token::Star) => {
                self.advance().ok();
                let expr = self.parse_prefix()?;
                let end = self.span().end;
                Ok(Expr::UnaryOp {
                    op: UnaryOp::Deref,
                    expr: Box::new(expr),
                    span: Span::new(start, end),
                })
            }
            Ok(Token::Move) => {
                self.advance().ok();
                let expr = self.parse_prefix()?;
                let end = self.span().end;
                Ok(Expr::Move(Box::new(expr), Span::new(start, end)))
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
                    let num_part = &s[2..]
                        .split(|c: char| c == 'i' || c == 'u')
                        .next()
                        .unwrap()
                        .replace('_', "");
                    i64::from_str_radix(&num_part, 16).unwrap_or(0)
                } else if s.starts_with("0b") || s.starts_with("0B") {
                    let num_part = &s[2..]
                        .split(|c: char| c == 'i' || c == 'u')
                        .next()
                        .unwrap()
                        .replace('_', "");
                    i64::from_str_radix(&num_part, 2).unwrap_or(0)
                } else {
                    let num_part = s
                        .split(|c: char| c == 'i' || c == 'u')
                        .next()
                        .unwrap()
                        .replace('_', "");
                    num_part.parse::<i64>().unwrap_or(0)
                };
                Ok(Expr::Literal(Literal::Int(value), Span::new(start, end)))
            }
            _ => Err(Diagnostic {
                message: "expected expression".to_string(),
                span: self.span(),
            }),
        }
    }

    fn parse_closure(&mut self, start: usize) -> Result<Expr, Diagnostic> {
        self.advance().ok();
        let mut params = Vec::new();
        loop {
            if matches!(self.peek(), Ok(Token::Pipe)) {
                self.advance().ok();
                break;
            }
            let name = match self.advance() {
                Ok(Token::Ident(n)) => n,
                Ok(tok) => {
                    return Err(Diagnostic {
                        message: format!("expected parameter name, found {:?}", tok),
                        span: self.span(),
                    });
                }
                Err(()) => {
                    return Err(Diagnostic {
                        message: "unexpected end of file in closure".to_string(),
                        span: self.span(),
                    });
                }
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
        self.expect(Token::LBrace)?;
        let body =
            self.with_restrictions(ParseRestrictions::VALUE_BLOCK, |this| this.parse_block())?;
        self.expect(Token::RBrace)?;
        let end = self.span().end;
        Ok(Expr::Closure {
            params,
            return_type,
            captures: Vec::new(),
            body,
            span: Span::new(start, end),
        })
    }

    fn parse_path_or_literal(&mut self, start: usize) -> Result<Expr, Diagnostic> {
        let mut path = Vec::new();
        let name = match self.advance() {
            Ok(Token::Ident(n)) => n,
            _ => unreachable!(),
        };
        path.push(name);
        while matches!(self.peek(), Ok(Token::ColonColon)) {
            self.advance().ok();
            if let Ok(Token::Ident(part)) = self.advance() {
                path.push(part);
            } else {
                return Err(Diagnostic {
                    message: "expected identifier after '::'".to_string(),
                    span: self.span(),
                });
            }
        }
        let restrict = self
            .restrictions
            .contains(ParseRestrictions::NO_STRUCT_LITERAL);
        match self.peek() {
            Ok(Token::LBrace) if !restrict => self.parse_struct_lit(path, start),
            Ok(Token::LParen) => {
                if path.len() >= 2 {
                    let variant = path.pop().unwrap();
                    self.parse_enum_lit(path, variant, start)
                } else {
                    self.parse_call(
                        Expr::Ident(
                            path.into_iter().next().unwrap(),
                            Span::new(start, self.span().start),
                        ),
                        start,
                    )
                }
            }
            _ => {
                if path.len() >= 2 {
                    let variant = path.pop().unwrap();
                    let end = self.span().end;
                    Ok(Expr::EnumLit {
                        path,
                        variant,
                        payload: None,
                        span: Span::new(start, end),
                    })
                } else {
                    Ok(Expr::Ident(
                        path.into_iter().next().unwrap(),
                        Span::new(start, self.span().end),
                    ))
                }
            }
        }
    }

    fn parse_struct_lit(&mut self, path: Vec<String>, start: usize) -> Result<Expr, Diagnostic> {
        self.expect(Token::LBrace)?;
        let mut fields = Vec::new();
        loop {
            if matches!(self.peek(), Ok(Token::RBrace)) {
                self.advance().ok();
                break;
            }
            let field_name = match self.advance() {
                Ok(Token::Ident(n)) => n,
                Ok(tok) => {
                    return Err(Diagnostic {
                        message: format!("expected field name, found {:?}", tok),
                        span: self.span(),
                    });
                }
                Err(()) => {
                    return Err(Diagnostic {
                        message: "unexpected end of file in struct literal".to_string(),
                        span: self.span(),
                    });
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
        path: Vec<String>,
        variant: String,
        start: usize,
    ) -> Result<Expr, Diagnostic> {
        self.expect(Token::LParen)?;
        let payload = self.parse_expr()?;
        self.expect(Token::RParen)?;
        let end = self.span().end;
        Ok(Expr::EnumLit {
            path,
            variant,
            payload: Some(Box::new(payload)),
            span: Span::new(start, end),
        })
    }

    fn parse_call(&mut self, callee: Expr, start: usize) -> Result<Expr, Diagnostic> {
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
            callee: Box::new(callee),
            args,
            comptime: false,
            span: Span::new(start, end),
        })
    }

    fn parse_literal(&mut self) -> Result<Expr, Diagnostic> {
        let start = self.span().start;
        let token = self.advance().unwrap();
        let end = self.span().end;
        let span = Span::new(start, end);
        match token {
            Token::IntLiteral(res) => match res {
                Ok(v) => Ok(Expr::Literal(Literal::Int(v), span)),
                Err(_) => Ok(Expr::Error(span)),
            },
            Token::HexLiteral(res) => match res {
                Ok(v) => Ok(Expr::Literal(Literal::Int(v), span)),
                Err(_) => Ok(Expr::Error(span)),
            },
            Token::BinLiteral(res) => match res {
                Ok(v) => Ok(Expr::Literal(Literal::Int(v), span)),
                Err(_) => Ok(Expr::Error(span)),
            },
            Token::FloatLiteral(res) => match res {
                Ok(v) => Ok(Expr::Literal(Literal::Float(v), span)),
                Err(_) => Ok(Expr::Error(span)),
            },
            Token::CharLiteral(res) => match res {
                Ok(v) => Ok(Expr::Literal(Literal::Char(v), span)),
                Err(_) => Ok(Expr::Error(span)),
            },
            Token::StringLiteral(res) => match res {
                Ok(s) => Ok(Expr::Literal(Literal::String(s), span)),
                Err(_) => Ok(Expr::Error(span)),
            },
            Token::ByteStringLiteral(res) => match res {
                Ok(b) => Ok(Expr::Literal(Literal::ByteString(b), span)),
                Err(_) => Ok(Expr::Error(span)),
            },
            Token::True => Ok(Expr::Literal(Literal::Bool(true), span)),
            Token::False => Ok(Expr::Literal(Literal::Bool(false), span)),
            _ => unreachable!(),
        }
    }

    fn parse_infix(&mut self, lhs: Expr, bp: u8) -> Result<Expr, Diagnostic> {
        let start = self.span().start;
        match self.peek() {
            Ok(Token::Bang) => {
                if matches!(lhs, Expr::Ident(..)) && matches!(self.peek_next(), Some(Token::LParen))
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
                        callee: Box::new(lhs),
                        args,
                        comptime: true,
                        span: Span::new(start, self.span().end),
                    })
                } else {
                    Err(Diagnostic {
                        message: "unexpected !".to_string(),
                        span: self.span(),
                    })
                }
            }
            Ok(Token::Question) => {
                self.advance().ok();
                let end = self.span().end;
                Ok(Expr::Try {
                    expr: Box::new(lhs),
                    span: Span::new(start, end),
                })
            }
            Ok(Token::As) => {
                self.advance().ok();
                let ty = self.parse_type()?;
                let end = self.span().end;
                Ok(Expr::Cast {
                    expr: Box::new(lhs),
                    ty: Box::new(ty),
                    safe: true,
                    span: Span::new(start, end),
                })
            }
            Ok(Token::Plus) => {
                self.advance().ok();
                self.binary(lhs, BinOp::Add, bp, start)
            }
            Ok(Token::Minus) => {
                self.advance().ok();
                self.binary(lhs, BinOp::Sub, bp, start)
            }
            Ok(Token::Star) => {
                self.advance().ok();
                self.binary(lhs, BinOp::Mul, bp, start)
            }
            Ok(Token::Slash) => {
                self.advance().ok();
                self.binary(lhs, BinOp::Div, bp, start)
            }
            Ok(Token::Percent) => {
                self.advance().ok();
                self.binary(lhs, BinOp::Rem, bp, start)
            }
            Ok(Token::PlusWrap) => {
                self.advance().ok();
                self.binary(lhs, BinOp::AddWrap, bp, start)
            }
            Ok(Token::MinusWrap) => {
                self.advance().ok();
                self.binary(lhs, BinOp::SubWrap, bp, start)
            }
            Ok(Token::StarWrap) => {
                self.advance().ok();
                self.binary(lhs, BinOp::MulWrap, bp, start)
            }
            Ok(Token::PlusSaturate) => {
                self.advance().ok();
                self.binary(lhs, BinOp::AddSaturate, bp, start)
            }
            Ok(Token::MinusSaturate) => {
                self.advance().ok();
                self.binary(lhs, BinOp::SubSaturate, bp, start)
            }
            Ok(Token::StarSaturate) => {
                self.advance().ok();
                self.binary(lhs, BinOp::MulSaturate, bp, start)
            }
            Ok(Token::PlusTrap) => {
                self.advance().ok();
                self.binary(lhs, BinOp::AddTrap, bp, start)
            }
            Ok(Token::MinusTrap) => {
                self.advance().ok();
                self.binary(lhs, BinOp::SubTrap, bp, start)
            }
            Ok(Token::StarTrap) => {
                self.advance().ok();
                self.binary(lhs, BinOp::MulTrap, bp, start)
            }
            Ok(Token::Ampersand) => {
                self.advance().ok();
                self.binary(lhs, BinOp::BitAnd, bp, start)
            }
            Ok(Token::Pipe) => {
                self.advance().ok();
                self.binary(lhs, BinOp::BitOr, bp, start)
            }
            Ok(Token::Caret) => {
                self.advance().ok();
                self.binary(lhs, BinOp::BitXor, bp, start)
            }
            Ok(Token::Shl) => {
                self.advance().ok();
                self.binary(lhs, BinOp::Shl, bp, start)
            }
            Ok(Token::Shr) => {
                self.advance().ok();
                self.binary(lhs, BinOp::Shr, bp, start)
            }
            Ok(Token::EqEq) => {
                self.advance().ok();
                self.binary(lhs, BinOp::Eq, bp, start)
            }
            Ok(Token::Neq) => {
                self.advance().ok();
                self.binary(lhs, BinOp::Neq, bp, start)
            }
            Ok(Token::Lt) => {
                self.advance().ok();
                self.binary(lhs, BinOp::Lt, bp, start)
            }
            Ok(Token::Gt) => {
                self.advance().ok();
                self.binary(lhs, BinOp::Gt, bp, start)
            }
            Ok(Token::Le) => {
                self.advance().ok();
                self.binary(lhs, BinOp::Le, bp, start)
            }
            Ok(Token::Ge) => {
                self.advance().ok();
                self.binary(lhs, BinOp::Ge, bp, start)
            }
            Ok(Token::And) => {
                self.advance().ok();
                self.binary(lhs, BinOp::And, bp, start)
            }
            Ok(Token::Or) => {
                self.advance().ok();
                self.binary(lhs, BinOp::Or, bp, start)
            }
            Ok(Token::DotDot) => {
                self.advance().ok();
                let end = if !matches!(
                    self.peek(),
                    Ok(Token::Semicolon) | Ok(Token::RBrace) | Ok(Token::Comma) | Ok(Token::RParen)
                ) {
                    Some(Box::new(self.parse_expr_bp(0)?))
                } else {
                    None
                };
                Ok(Expr::Range {
                    start: Some(Box::new(lhs)),
                    end,
                    inclusive: false,
                    span: Span::new(start, self.span().end),
                })
            }
            Ok(Token::DotDotEq) => {
                self.advance().ok();
                let end = self.parse_expr_bp(0)?;
                Ok(Expr::Range {
                    start: Some(Box::new(lhs)),
                    end: Some(Box::new(end)),
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
                    callee: Box::new(lhs),
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
                    base: Box::new(lhs),
                    index: Box::new(index),
                    span: Span::new(start, self.span().end),
                })
            }
            Ok(Token::Dot) => {
                self.advance().ok();
                if let Ok(Token::Ident(field)) = self.advance() {
                    Ok(Expr::FieldAccess {
                        base: Box::new(lhs),
                        field,
                        span: Span::new(start, self.span().end),
                    })
                } else {
                    Err(Diagnostic {
                        message: "expected field name after '.'".to_string(),
                        span: self.span(),
                    })
                }
            }
            Ok(Token::Apostrophe) => {
                self.advance().ok();
                if let Ok(Token::Ident(attr)) = self.advance() {
                    Ok(Expr::AttrAccess {
                        base: Box::new(lhs),
                        attr,
                        span: Span::new(start, self.span().end),
                    })
                } else {
                    Err(Diagnostic {
                        message: "expected attribute name after '''".to_string(),
                        span: self.span(),
                    })
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
                            _ => {
                                return Err(Diagnostic {
                                    message: "expected binding name after 'as'".to_string(),
                                    span: self.span(),
                                });
                            }
                        }
                    } else {
                        None
                    };
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
                    expr: Box::new(lhs),
                    branches,
                    span: Span::new(start, self.span().end),
                })
            }
            _ => unreachable!(),
        }
    }

    fn binary(&mut self, lhs: Expr, op: BinOp, bp: u8, start: usize) -> Result<Expr, Diagnostic> {
        let rhs = self.parse_expr_bp(bp)?;
        Ok(Expr::BinaryOp {
            left: Box::new(lhs),
            op,
            right: Box::new(rhs),
            span: Span::new(start, self.span().end),
        })
    }

    fn peek_next(&mut self) -> Option<Token> {
        let mut lexer = self.lexer.clone();
        loop {
            match lexer.next() {
                Some(Ok(Token::WhitespaceOrComment)) => continue,
                Some(Ok(tok)) => return Some(tok),
                _ => return None,
            }
        }
    }

    fn parse_if_expr(&mut self) -> Result<Expr, Diagnostic> {
        let start = self.span().start;
        self.advance().ok();
        if matches!(self.peek(), Ok(Token::Let)) {
            self.advance().ok();
            let pattern = self.parse_pattern()?;
            self.expect(Token::Assign)?;
            let old_restrict = self.restrictions;
            self.restrictions |= ParseRestrictions::NO_STRUCT_LITERAL;
            let scrutinee = self.parse_expr()?;
            self.restrictions = old_restrict;
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
                scrutinee: Box::new(scrutinee),
                then_branch,
                else_branch,
                span: Span::new(start, end),
            });
        }
        let old_restrict = self.restrictions;
        self.restrictions |= ParseRestrictions::NO_STRUCT_LITERAL;
        let cond = self.parse_expr()?;
        self.restrictions = old_restrict;
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
            cond: Box::new(cond),
            then_branch,
            else_branch,
            is_expression: true,
            span: Span::new(start, end),
        })
    }

    fn parse_match_expr(&mut self) -> Result<Expr, Diagnostic> {
        let start = self.span().start;
        self.advance().ok();
        let old_restrict = self.restrictions;
        self.restrictions |= ParseRestrictions::NO_STRUCT_LITERAL;
        let scrutinee = self.parse_expr()?;
        self.restrictions = old_restrict;
        self.expect(Token::LBrace)?;
        let mut arms = Vec::new();
        loop {
            if matches!(self.peek(), Ok(Token::RBrace)) {
                self.advance().ok();
                break;
            }
            let arm_start = self.span().start;
            let pattern = self.parse_pattern()?;
            let guard = if matches!(self.peek(), Ok(Token::If)) {
                self.advance().ok();
                Some(Box::new(self.parse_expr()?))
            } else {
                None
            };
            self.expect(Token::FatArrow)?;
            let body =
                self.with_restrictions(ParseRestrictions::VALUE_BLOCK | ParseRestrictions::NO_STRUCT_LITERAL, |this| this.parse_expr())?;
            arms.push(MatchArm {
                pattern,
                guard,
                body,
                span: Span::new(arm_start, self.span().end),
            });
            if matches!(self.peek(), Ok(Token::Comma)) {
                self.advance().ok();
            } else {
                self.expect(Token::RBrace)?;
                break;
            }
        }
        let end = self.span().end;
        Ok(Expr::Match {
            scrutinee: Box::new(scrutinee),
            arms,
            span: Span::new(start, end),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_parse(source: &str) -> Program {
        let mut parser = Parser::new(source);
        match parser.parse_program() {
            Ok(prog) => {
                assert!(
                    parser.diagnostics.is_empty(),
                    "unexpected diagnostics: {:?}",
                    parser.diagnostics
                );
                prog
            }
            Err(diags) => panic!("parse failed with diagnostics: {:?}", diags),
        }
    }

    #[test]
    fn test_empty_function() {
        let program = check_parse("def main() { }");
        assert_eq!(program.items.len(), 1);
        match &program.items[0] {
            Stmt::FunctionDef {
                name, params, body, ..
            } => {
                assert_eq!(name, "main");
                assert!(params.is_empty());
                assert!(body.as_ref().map_or(false, |b| b.is_empty()));
            }
            _ => panic!("expected FunctionDef"),
        }
    }

    #[test]
    fn test_variable_def() {
        let program = check_parse("def main() { set x = 42; }");
        match &program.items[0] {
            Stmt::FunctionDef { body, .. } => match &body.as_ref().unwrap()[0] {
                Stmt::VariableDef { name, .. } => assert_eq!(name.as_ref().unwrap(), "x"),
                _ => panic!("expected VariableDef"),
            },
            _ => panic!("expected FunctionDef"),
        }
    }

    #[test]
    fn test_if_stmt() {
        let program = check_parse("def main() { if true { } }");
        match &program.items[0] {
            Stmt::FunctionDef { body, .. } => {
                assert!(matches!(body.as_ref().unwrap()[0], Stmt::If { .. }));
            }
            _ => panic!("expected FunctionDef"),
        }
    }

    #[test]
    fn test_scope_cleanup_with_at() {
        let program = check_parse("def main() { scope_cleanup @close_file { } }");
        match &program.items[0] {
            Stmt::FunctionDef { body, .. } => match &body.as_ref().unwrap()[0] {
                Stmt::ScopeCleanup { name, body: _, .. } => assert_eq!(name, "close_file"),
                _ => panic!("expected ScopeCleanup"),
            },
            _ => panic!("expected FunctionDef"),
        }
    }

    #[test]
    fn test_trigger_with_at() {
        let program = check_parse("def main() { trigger @close_file; }");
        match &program.items[0] {
            Stmt::FunctionDef { body, .. } => match &body.as_ref().unwrap()[0] {
                Stmt::Trigger { name, .. } => assert_eq!(name, "close_file"),
                _ => panic!("expected Trigger"),
            },
            _ => panic!("expected FunctionDef"),
        }
    }

    #[test]
    fn test_reference_type() {
        let program = check_parse("def main() { set x: &Int<32> = 0; }");
        assert!(program.items.len() == 1);
    }

    #[test]
    fn test_pointer_type() {
        let program = check_parse("def main() { set x: *Int<32> = 0; }");
        assert!(program.items.len() == 1);
    }

    #[test]
    fn test_slice_type() {
        let program = check_parse("def main() { set x: [Int<32>] = 0; }");
        assert!(program.items.len() == 1);
    }

    #[test]
    fn test_array_type() {
        let program = check_parse("def main() { set x: [Int<32>; 10] = 0; }");
        assert!(program.items.len() == 1);
    }

    #[test]
    fn test_dyn_trait_type() {
        let program = check_parse("def main() { set x: dyn Display = 0; }");
        assert!(program.items.len() == 1);
    }

    #[test]
    fn test_exists_type() {
        let program = check_parse("type Age = exists n: UInt<8> invariant n >= 18;");
        assert!(program.items.len() == 1);
    }

    #[test]
    fn test_ellipsis_is_invalid() {
        let src = "def main() { ...; }";
        let mut parser = Parser::new(src);
        let result = parser.parse_program();
        assert!(result.is_err() || !parser.diagnostics.is_empty());
    }

    #[test]
    fn test_struct_literal() {
        let program = check_parse("def main() { set e = Employee { id = 1, name = b\"Alice\" }; }");
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_enum_literal() {
        let program = check_parse("def main() { set d = Department::Engineering; }");
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_move_expression() {
        let program = check_parse("def main() { set x = 1; set y = move x; }");
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_suffixed_literal() {
        let program = check_parse("def main() { set x = 42i32; }");
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_unsafe_block() {
        let program = check_parse("def main() { unsafe { } }");
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_try_expression() {
        let program = check_parse(
            "def main() -> Result<(), Error> { let x = do_something()?; return Ok(()); }",
        );
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_cast() {
        let program = check_parse("def main() { set x = 42 as Float<64>; }");
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_ref_prefix() {
        let program = check_parse("def main() { set x = &y; }");
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_deref_prefix() {
        let program = check_parse("def main() { set x = *y; }");
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_finally_block() {
        let program = check_parse("def main() { } finally { }");
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_impl_block() {
        let program = check_parse("impl Drop for UniqueToken { def drop(&mut self) { } }");
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_closure() {
        let program =
            check_parse("def main() { set f = |x: Int<32>, y: Int<32>| -> Int<32> { x + y; }; }");
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_pattern_struct() {
        let program = check_parse("def main() { let Point { x, y } = p; }");
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_pattern_enum() {
        let program = check_parse("def main() { let Some(v) = opt; }");
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_pattern_literal() {
        let program = check_parse("def main() { match x { 1 => {}, _ => {} }; }");
        assert_eq!(program.items.len(), 1);
    }

    #[test]
    fn test_ghost_variable() {
        let program = check_parse("def main() { ghost set mut x = 0; }");
        assert_eq!(program.items.len(), 1);
    }
}
