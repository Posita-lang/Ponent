//! Parsing of type definitions: `type Name = ...` including struct, enum, alias, opaque, and layout aliases.

use super::*;
use crate::ast::*;
use crate::diagnostics::Diagnostic;
use crate::lexer::Token;
use crate::symbol::Symbol;

impl<'input> Parser<'input> {
    // -------------------------------------------------------------------------
    // Main `type` definition entry
    // -------------------------------------------------------------------------

    pub(super) fn parse_type_def(
        &mut self,
        mut attributes: Vec<Attribute<'input>>,
        doc: Option<String>,
    ) -> Result<Stmt<'input>, Diagnostic> {
        let start = self.span().start;
        self.advance().ok();
        let name = match self.advance() {
            Ok(Token::Ident(name)) => name,
            _ => return Err(Diagnostic::error("expected type name")
                .with_code_str("E004")
                .with_help(
                    "after `type`, a type name (identifier) is expected — e.g. `type MyType = ...`",
                )
                .with_suggestion("add a type name after `type`, e.g. `type MyType = Int<32>;`")
                .with_span(self.span())),
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
        let mut ty = if let Ok(Token::Ident(s)) = self.peek().clone() {
            if s.eq_str("struct") {
                self.advance().ok();
                return self.parse_struct_def(start, name, params, attributes, doc);
            } else if s.eq_str("impl") {
                self.advance().ok();
                return self.parse_opaque_def(start, name, params, attributes, doc);
            } else if s.eq_str("enum") {
                self.advance().ok();
                return self.parse_enum_def(start, name, params, attributes, doc);
            } else {
                self.parse_type()?
            }
        } else {
            self.parse_type()?
        };

        if matches!(self.peek(), Ok(Token::Where)) {
            self.advance().ok();
            let invariant = self.parse_expr()?;
            ty = Type::WhereShorthand {
                base: Self::alloc_shared(self.arena, ty),
                invariant: Self::alloc_shared(self.arena, invariant),
                span: Span::new(start, self.span().end),
            };
        }

        if matches!(self.peek(), Ok(Token::Pipe)) {
            let mut types = vec![ty];
            while matches!(self.peek(), Ok(Token::Pipe)) {
                self.advance().ok();
                types.push(self.parse_type()?);
            }
            ty = Type::Union(types, Span::new(start, self.span().end));
        }

        let modifiers = self.parse_type_modifiers()?;
        if matches!(self.peek(), Ok(Token::Semicolon)) {
            self.advance().ok();
        }
        let end = self.span().end;
        Ok(Stmt::TypeDef {
            span: Span::new(start, end),
            attributes,
            doc,
            name,
            params,
            definition: TypeDefinition::Alias(ty, modifiers),
            contracts: Vec::new(),
        })
    }

    // -------------------------------------------------------------------------
    // Struct definition
    // -------------------------------------------------------------------------

    fn parse_struct_def(
        &mut self,
        start: usize,
        name: Symbol,
        params: Vec<TypeParam<'input>>,
        attributes: Vec<Attribute<'input>>,
        doc: Option<String>,
    ) -> Result<Stmt<'input>, Diagnostic> {
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
                    return Err(Diagnostic::error("expected field name")
                        .with_code_str("E004")
                        .with_help("struct fields must have a name — e.g. `name: String`")
                        .with_suggestion("add a field name like `name`, `age`, or `value`")
                        .with_span(self.span()));
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
        let modifiers = self.parse_type_modifiers()?;
        let end = self.span().end;
        Ok(Stmt::TypeDef {
            span: Span::new(start, end),
            attributes,
            doc,
            name,
            params,
            definition: TypeDefinition::Struct(fields, modifiers),
            contracts: Vec::new(),
        })
    }

    // -------------------------------------------------------------------------
    // Opaque (TAIT) definition
    // -------------------------------------------------------------------------

    fn parse_opaque_def(
        &mut self,
        start: usize,
        name: Symbol,
        params: Vec<TypeParam<'input>>,
        attributes: Vec<Attribute<'input>>,
        doc: Option<String>,
    ) -> Result<Stmt<'input>, Diagnostic> {
        let trait_ty = self.parse_type()?;
        let modifiers = self.parse_type_modifiers()?;
        if matches!(self.peek(), Ok(Token::Semicolon)) {
            self.advance().ok();
        }
        let end = self.span().end;
        Ok(Stmt::TypeDef {
            span: Span::new(start, end),
            attributes,
            doc,
            name,
            params,
            definition: TypeDefinition::Opaque(trait_ty, modifiers),
            contracts: Vec::new(),
        })
    }

    // -------------------------------------------------------------------------
    // Enum definition (including GADT)
    // -------------------------------------------------------------------------

    fn parse_enum_def(
        &mut self,
        start: usize,
        name: Symbol,
        params: Vec<TypeParam<'input>>,
        attributes: Vec<Attribute<'input>>,
        doc: Option<String>,
    ) -> Result<Stmt<'input>, Diagnostic> {
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
                    return Err(Diagnostic::error("expected variant name")
                        .with_code_str("E004")
                        .with_help("enum variants must have a name — e.g. `enum { A, B }`")
                        .with_suggestion("add a variant name like `VariantA`, `None`, or `Some`")
                        .with_span(self.span()));
                }
            };
            let mut variant_exists_params = Vec::new();
            let payload = if matches!(self.peek(), Ok(Token::LParen)) {
                self.advance().ok();
                if matches!(self.peek(), Ok(Token::Exists)) {
                    self.advance().ok();
                    loop {
                        match self.advance() {
                            Ok(Token::Ident(name)) => variant_exists_params.push(name),
                            Ok(tok) => {
                                return Err(Diagnostic::error(format!(
                                    "expected type variable name in `exists` clause, found {:?}",
                                    tok
                                ))
                                .with_span(self.span()));
                            }
                            Err(()) => {
                                return Err(Diagnostic::error(
                                    "unexpected end of file in `exists` clause",
                                )
                                .with_span(self.span()));
                            }
                        }
                        match self.peek() {
                            Ok(Token::Comma) => {
                                self.advance().ok();
                                continue;
                            }
                            Ok(Token::Colon) => {
                                self.advance().ok();
                                break;
                            }
                            _ => {
                                return Err(Diagnostic::error(
                                    "expected `:` after `exists` variable list",
                                )
                                .with_help("use `exists X: Type` or `exists X, Y: Type` syntax")
                                .with_span(self.span()));
                            }
                        }
                    }
                }
                let ty = self.parse_type()?;
                if !variant_exists_params.is_empty() && matches!(self.peek(), Ok(Token::Comma)) {
                    return Err(Diagnostic::error("expected `)` after the `exists` variable type")
                        .with_help("multiple `exists` variables share ONE type — use `exists X, Y: Type` (comma-separated names, then a single `:` and one type), not `exists X: T, exists Y: T`")
                        .with_span(self.span()));
                }
                self.expect(Token::RParen)?;
                Some(ty)
            } else {
                None
            };
            let eq_spec = if matches!(self.peek(), Ok(Token::When)) {
                self.advance().ok();
                self.parse_gadt_constraints()?
            } else {
                Vec::new()
            };
            variants.push(EnumVariant {
                name: v_name,
                payload,
                eq_spec,
                exists_params: variant_exists_params,
                span: Span::new(start, self.span().end),
            });
            if matches!(self.peek(), Ok(Token::Comma)) {
                self.advance().ok();
            } else {
                self.expect(Token::RBrace)?;
                break;
            }
        }
        let missing_match = if matches!(self.peek(), Ok(Token::With))
            && matches!(self.peek_next(), Some(Token::MissingMatch))
        {
            self.advance().ok();
            self.expect(Token::MissingMatch)?;
            self.expect(Token::Assign)?;
            let msg = match self.advance() {
                Ok(Token::StringLiteral(Ok(s))) => s,
                _ => return Err(Diagnostic::error("expected string for missing_match")
                    .with_code_str("E004")
                    .with_help("`missing_match` expects a string literal — e.g. `with missing_match = \"message\"`")
                    .with_suggestion("use a string literal like `\"not all variants covered\"`")
                    .with_span(self.span())),
            };
            self.expect(Token::Semicolon)?;
            Some(msg)
        } else {
            None
        };
        let enum_modifiers = self.parse_type_modifiers()?;
        let end = self.span().end;
        Ok(Stmt::TypeDef {
            span: Span::new(start, end),
            attributes,
            doc,
            name,
            params,
            definition: TypeDefinition::Enum(variants, missing_match, enum_modifiers),
            contracts: Vec::new(),
        })
    }

    // -------------------------------------------------------------------------
    // GADT constraints: `Param == ConcreteType [and ...]`
    // -------------------------------------------------------------------------

    pub(super) fn parse_gadt_constraints(
        &mut self,
    ) -> Result<Vec<(Symbol, Type<'input>)>, Diagnostic> {
        let mut constraints = Vec::new();
        loop {
            let param_name = match self.advance() {
                Ok(Token::Ident(name)) => name,
                Ok(tok) => {
                    return Err(Diagnostic::error(format!(
                        "expected type parameter name in GADT constraint, found {:?}",
                        tok
                    ))
                    .with_code_str("E004")
                    .with_help("GADT constraints use `Param == ConcreteType` syntax")
                    .with_suggestion("use `T == Int<32>` where `T` is a type parameter of the enum")
                    .with_span(self.span()));
                }
                Err(()) => {
                    return Err(
                        Diagnostic::error("unexpected end of file in GADT constraint")
                            .with_code_str("E002")
                            .with_help("GADT constraints use `Param == ConcreteType` syntax")
                            .with_span(self.span()),
                    );
                }
            };
            self.expect(Token::EqEq)?;
            let concrete_ty = self.parse_type()?;
            constraints.push((param_name, concrete_ty));
            match self.peek() {
                Ok(Token::And) => {
                    self.advance().ok();
                    continue;
                }
                _ => break,
            }
        }
        Ok(constraints)
    }

    // -------------------------------------------------------------------------
    // Type modifiers (`with default = ...`, `with overflow = ...`, etc.)
    // -------------------------------------------------------------------------

    pub(super) fn parse_type_modifiers(&mut self) -> Result<Vec<TypeModifier<'input>>, Diagnostic> {
        let mut modifiers = Vec::new();
        while matches!(self.peek(), Ok(Token::With)) {
            self.advance().ok();
            match self.peek() {
                Ok(Token::Ident(_)) | Ok(Token::Default) | Ok(Token::NoDefault) => {
                    let tok = self.advance().map_err(|_| Diagnostic::error("unexpected token")
                        .with_code_str("E003")
                        .with_help("expected a type modifier name (`overflow`, `validate`, `default`, `no_default`) after `with`")
                        .with_span(Span::new(0, 0)))?;
                    match tok {
                        Token::Ident(ref s) if s.eq_str("overflow") => {
                            self.expect(Token::Assign)?;
                            let policy = match self.advance() {
                                Ok(Token::Wrap) => OverflowPolicy::Wrap,
                                Ok(Token::Saturate) => OverflowPolicy::Saturate,
                                Ok(Token::Trap) => OverflowPolicy::Trap,
                                Ok(Token::Ieee) => OverflowPolicy::Ieee,
                                _ => return Err(Diagnostic::error("expected overflow policy (wrap, saturate, trap, ieee)")
                                    .with_code_str("E007")
                                    .with_help("`overflow` policy must be one of: `wrap`, `saturate`, `trap`, or `ieee` (floats only)")
                                    .with_suggestion("use one of: `wrap`, `saturate`, `trap`, or `ieee`")
                                    .with_span(self.span())),
                            };
                            modifiers.push(TypeModifier::Overflow(policy));
                            if matches!(self.peek(), Ok(Token::Semicolon)) {
                                self.advance().ok();
                            }
                        }
                        Token::Ident(ref s) if s.eq_str("validate") => {
                            self.expect(Token::Assign)?;
                            let closure = self.parse_closure(self.span().start)?;
                            modifiers.push(TypeModifier::Validate(closure));
                            if matches!(self.peek(), Ok(Token::Semicolon)) {
                                self.advance().ok();
                            }
                        }
                        Token::Default => {
                            self.expect(Token::Assign)?;
                            let expr = self.parse_expr()?;
                            modifiers.push(TypeModifier::Default(expr));
                            if matches!(self.peek(), Ok(Token::Semicolon)) {
                                self.advance().ok();
                            }
                        }
                        Token::NoDefault => {
                            modifiers.push(TypeModifier::NoDefault);
                            if matches!(self.peek(), Ok(Token::Semicolon)) {
                                self.advance().ok();
                            }
                        }
                        _ => {
                            while !matches!(
                                self.peek(),
                                Ok(Token::Semicolon) | Ok(Token::RBrace) | Err(())
                            ) {
                                self.advance().ok();
                            }
                            if matches!(self.peek(), Ok(Token::Semicolon)) {
                                self.advance().ok();
                            }
                        }
                    }
                }
                _ => {
                    while !matches!(
                        self.peek(),
                        Ok(Token::Semicolon) | Ok(Token::RBrace) | Err(())
                    ) {
                        self.advance().ok();
                    }
                    if matches!(self.peek(), Ok(Token::Semicolon)) {
                        self.advance().ok();
                    }
                }
            }
        }
        Ok(modifiers)
    }

    // -------------------------------------------------------------------------
    // Layout alias definition
    // -------------------------------------------------------------------------

    pub(super) fn parse_layout_def(
        &mut self,
        mut attributes: Vec<Attribute<'input>>,
    ) -> Result<Stmt<'input>, Diagnostic> {
        let start = self.span().start;
        let name = match self.advance() {
            Ok(Token::Ident(name)) => name,
            Ok(tok) => self
                .keyword_to_ident(&tok)
                .unwrap_or_else(|| Symbol::intern(&format!("{:?}", tok))),
            Err(()) => {
                return Err(
                    Diagnostic::error("unexpected end of file in layout definition")
                        .with_code_str("E002")
                        .with_span(self.span()),
                );
            }
        };
        self.expect(Token::LBrace)?;
        loop {
            if matches!(self.peek(), Ok(Token::Comma)) {
                self.advance().ok();
                continue;
            }
            if matches!(self.peek(), Ok(Token::Semicolon)) {
                self.advance().ok();
                break;
            }
            let attr_name = match self.advance() {
                Ok(Token::Ident(name)) => name,
                Ok(tok) => self
                    .keyword_to_ident(&tok)
                    .unwrap_or_else(|| Symbol::intern(&format!("{:?}", tok))),
                Err(()) => break,
            };
            let mut args = Vec::new();
            if matches!(self.peek(), Ok(Token::LParen)) {
                self.advance().ok();
                let arg_name = match self.advance() {
                    Ok(Token::Ident(name)) => name,
                    Ok(tok) => self
                        .keyword_to_ident(&tok)
                        .unwrap_or_else(|| Symbol::intern(&format!("{:?}", tok))),
                    Err(()) => Symbol::INVALID,
                };
                args.push(crate::ast::Expr::Ident(arg_name, self.span()));
                self.expect(Token::RParen)?;
            }
            attributes.push(Attribute {
                name: attr_name,
                args,
                named_args: Vec::new(),
                span: Span::new(start, self.span().end),
            });
        }
        self.expect(Token::RBrace)?;
        let end = self.span().end;
        Ok(Stmt::LayoutDef {
            name,
            attributes,
            span: Span::new(start, end),
        })
    }

    // -------------------------------------------------------------------------
    // Type capture parameters for `set auto<...>`
    // -------------------------------------------------------------------------

    pub(super) fn parse_type_capture_params(
        &mut self,
    ) -> Result<Vec<TypeParam<'input>>, Diagnostic> {
        self.advance().ok();
        let mut params = Vec::new();
        loop {
            match self.peek() {
                Ok(Token::Gt) | Ok(Token::Shr) => {
                    self.expect_gt()?;
                    break;
                }
                Ok(Token::Ident(name)) => {
                    let name = *name;
                    self.advance().ok();
                    params.push(TypeParam {
                        name,
                        bounds: vec![],
                        kind: TypeParamKind::Type,
                        span: self.span(),
                    });
                    if matches!(self.peek(), Ok(Token::Comma)) {
                        self.advance().ok();
                    } else if matches!(self.peek(), Ok(Token::Gt) | Ok(Token::Shr)) {
                        self.expect_gt()?;
                        break;
                    } else {
                        return Err(Diagnostic::error(
                            "expected `,` or `>` in capture parameter list",
                        )
                        .with_span(self.span()));
                    }
                }
                _ => {
                    return Err(Diagnostic::error(
                        "expected capture parameter name or `>` in `set auto<...>`",
                    )
                    .with_span(self.span()));
                }
            }
        }
        Ok(params)
    }
}
