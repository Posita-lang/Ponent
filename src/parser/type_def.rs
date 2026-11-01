//! Parsing of type definitions: `type Name = ...` including struct, enum, alias, opaque, and layout aliases.

use super::*;
use crate::ast::*;
use crate::diagnostics::Diagnostic;
use crate::diagnostics::ErrCode;
use crate::lexer::Token;
use crate::lexer::literal::parse_string_literal;
use crate::symbol::Symbol;

impl<'src, 'input> Parser<'src, 'input> {
    // -------------------------------------------------------------------------
    // Main `type` definition entry
    // -------------------------------------------------------------------------

    pub(super) fn parse_type_def(
        &mut self,
        attributes: Vec<Attribute<'input>>,
        doc: Option<String>,
        bump: &'input bumpalo::Bump,
    ) -> Result<Item<'input>, Diagnostic<'src>> {
        let start = self.span().start;
        self.advance();
        let name = match self.advance() {
            Some(Token::Ident(name)) => name,
            _ => return Err(Diagnostic::error("expected type name")
                .with_code_str(ErrCode::E004)
                .with_help(
                    "after `type`, a type name (identifier) is expected — e.g. `type MyType = ...`",
                )
                .with_suggestion("add a type name after `type`, e.g. `type MyType = Int<32>;`")
                .with_span(self.span())),
        };
        let params = if self
            .restrictions
            .contains(ParseRestrictions::ALLOW_TYPE_PARAMS)
            && matches!(self.peek(), Some(Token::Lt))
        {
            self.parse_type_params(bump)?
        } else {
            Vec::new()
        };
        self.expect(Token::Assign)?;
        let mut ty = if let Some(Token::Ident(s)) = self.peek().clone() {
            if s.eq_str("struct") {
                self.advance();
                return self.parse_struct_def(start, name, params, attributes, doc, bump);
            } else if s.eq_str("impl") {
                self.advance();
                return self.parse_opaque_def(start, name, params, attributes, doc, bump);
            } else if s.eq_str("enum") {
                self.advance();
                return self.parse_enum_def(start, name, params, attributes, doc, bump);
            } else {
                self.parse_type(bump)?
            }
        } else {
            self.parse_type(bump)?
        };

        if matches!(self.peek(), Some(Token::Where)) {
            self.advance();
            let invariant = self.parse_expr(bump)?;
            ty = Type::WhereShorthand {
                base: Self::alloc_shared(bump, ty),
                invariant: Self::alloc_shared(bump, invariant),
                span: Span::new(self.file_id, start, self.span().end),
            };
        }

        if matches!(self.peek(), Some(Token::Pipe)) {
            let mut types = vec![ty];
            while matches!(self.peek(), Some(Token::Pipe)) {
                self.advance();
                types.push(self.parse_type(bump)?);
            }
            ty = Type::Union(
                bump.alloc_slice_clone(&types),
                Span::new(self.file_id, start, self.span().end),
            );
        }

        let modifiers = self.parse_type_modifiers(bump)?;
        if matches!(self.peek(), Some(Token::Semicolon)) {
            self.advance();
        }
        let end = self.span().end;
        Ok(Item::TypeDef {
            span: Span::new(self.file_id, start, end),
            attributes: bump.alloc_slice_clone(&attributes),
            doc,
            name,
            params: bump.alloc_slice_clone(&params),
            definition: TypeDefinition::Alias(ty, bump.alloc_slice_clone(&modifiers)),
            contracts: &[],
        })
    }

    // -------------------------------------------------------------------------
    // Struct definition
    // -------------------------------------------------------------------------

    fn parse_struct_def(
        &mut self,
        start: u32,
        name: Symbol,
        params: Vec<TypeParam<'input>>,
        attributes: Vec<Attribute<'input>>,
        doc: Option<String>,
        bump: &'input bumpalo::Bump,
    ) -> Result<Item<'input>, Diagnostic<'src>> {
        self.expect(Token::LBrace)?;
        let mut fields = Vec::new();
        loop {
            if matches!(self.peek(), Some(Token::RBrace)) {
                self.advance();
                break;
            }
            let field_name = match self.advance() {
                Some(Token::Ident(n)) => n,
                _ => {
                    return Err(Diagnostic::error("expected field name")
                        .with_code_str(ErrCode::E004)
                        .with_help("struct fields must have a name — e.g. `name: String`")
                        .with_suggestion("add a field name like `name`, `age`, or `value`")
                        .with_span(self.span()));
                }
            };
            self.expect(Token::Colon)?;
            let field_ty = self.parse_type(bump)?;
            let default = if matches!(self.peek(), Some(Token::Assign)) {
                self.advance();
                Some(self.parse_expr(bump)?)
            } else {
                None
            };
            fields.push(StructField {
                name: field_name,
                ty: field_ty,
                default,
                span: Span::new(self.file_id, start, self.span().end),
            });
            if matches!(self.peek(), Some(Token::Comma)) {
                self.advance();
            } else {
                self.expect(Token::RBrace)?;
                break;
            }
        }
        let modifiers = self.parse_type_modifiers(bump)?;
        let end = self.span().end;
        Ok(Item::TypeDef {
            span: Span::new(self.file_id, start, end),
            attributes: bump.alloc_slice_clone(&attributes),
            doc,
            name,
            params: bump.alloc_slice_clone(&params),
            definition: TypeDefinition::Struct(
                bump.alloc_slice_clone(&fields),
                bump.alloc_slice_clone(&modifiers),
            ),
            contracts: &[],
        })
    }

    // -------------------------------------------------------------------------
    // Opaque (TAIT) definition
    // -------------------------------------------------------------------------

    fn parse_opaque_def(
        &mut self,
        start: u32,
        name: Symbol,
        params: Vec<TypeParam<'input>>,
        attributes: Vec<Attribute<'input>>,
        doc: Option<String>,
        bump: &'input bumpalo::Bump,
    ) -> Result<Item<'input>, Diagnostic<'src>> {
        let trait_ty = self.parse_type(bump)?;
        let modifiers = self.parse_type_modifiers(bump)?;
        if matches!(self.peek(), Some(Token::Semicolon)) {
            self.advance();
        }
        let end = self.span().end;
        Ok(Item::TypeDef {
            span: Span::new(self.file_id, start, end),
            attributes: bump.alloc_slice_clone(&attributes),
            doc,
            name,
            params: bump.alloc_slice_clone(&params),
            definition: TypeDefinition::Opaque(trait_ty, bump.alloc_slice_clone(&modifiers)),
            contracts: &[],
        })
    }

    // -------------------------------------------------------------------------
    // Enum definition (including GADT)
    // -------------------------------------------------------------------------

    fn parse_enum_def(
        &mut self,
        start: u32,
        name: Symbol,
        params: Vec<TypeParam<'input>>,
        attributes: Vec<Attribute<'input>>,
        doc: Option<String>,
        bump: &'input bumpalo::Bump,
    ) -> Result<Item<'input>, Diagnostic<'src>> {
        self.expect(Token::LBrace)?;
        let mut variants = Vec::new();
        loop {
            if matches!(self.peek(), Some(Token::RBrace)) {
                self.advance();
                break;
            }
            let v_name = match self.advance() {
                Some(Token::Ident(n)) => n,
                _ => {
                    return Err(Diagnostic::error("expected variant name")
                        .with_code_str(ErrCode::E004)
                        .with_help("enum variants must have a name — e.g. `enum { A, B }`")
                        .with_suggestion("add a variant name like `VariantA`, `None`, or `Some`")
                        .with_span(self.span()));
                }
            };
            let mut variant_exists_params = Vec::new();
            let payload = if matches!(self.peek(), Some(Token::LParen)) {
                self.advance();
                if matches!(self.peek(), Some(Token::Exists)) {
                    self.advance();
                    loop {
                        match self.advance() {
                            Some(Token::Ident(name)) => variant_exists_params.push(name),
                            Some(tok) => {
                                return Err(Diagnostic::error(format!(
                                    "expected type variable name in `exists` clause, found {:?}",
                                    tok
                                ))
                                .with_span(self.span()));
                            }
                            None => {
                                return Err(Diagnostic::error(
                                    "unexpected end of file in `exists` clause",
                                )
                                .with_span(self.span()));
                            }
                        }
                        match self.peek() {
                            Some(Token::Comma) => {
                                self.advance();
                                continue;
                            }
                            Some(Token::Colon) => {
                                self.advance();
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
                let ty = self.parse_type(bump)?;
                if !variant_exists_params.is_empty() && matches!(self.peek(), Some(Token::Comma)) {
                    return Err(Diagnostic::error("expected `)` after the `exists` variable type")
                        .with_help("multiple `exists` variables share ONE type — use `exists X, Y: Type` (comma-separated names, then a single `:` and one type), not `exists X: T, exists Y: T`")
                        .with_span(self.span()));
                }
                self.expect(Token::RParen)?;
                Some(ty)
            } else {
                None
            };
            let eq_spec = if matches!(self.peek(), Some(Token::When)) {
                self.advance();
                self.parse_gadt_constraints(bump)?
            } else {
                Vec::new()
            };
            variants.push(EnumVariant {
                name: v_name,
                payload,
                eq_spec,
                exists_params: variant_exists_params,
                span: Span::new(self.file_id, start, self.span().end),
            });
            if matches!(self.peek(), Some(Token::Comma)) {
                self.advance();
            } else {
                self.expect(Token::RBrace)?;
                break;
            }
        }
        let missing_match = if matches!(self.peek(), Some(Token::With))
            && matches!(self.peek_next(), Some(Token::MissingMatch))
        {
            self.advance();
            self.expect(Token::MissingMatch)?;
            self.expect(Token::Assign)?;
            let msg = match self.advance() {
                // The token carries the raw slice; parse at consumption time.
                Some(Token::StringLiteral(s)) => match parse_string_literal(s) {
                    Ok(c) => c.into_owned(),
                    Err(e) => {
                        return Err(Diagnostic::error(format!(
                            "invalid string for missing_match: {}",
                            e
                        ))
                        .with_code_str(ErrCode::E004)
                        .with_help(
                            "`missing_match` expects a string literal — e.g. `with missing_match = \"message\"`",
                        )
                        .with_suggestion("use a string literal like `\"not all variants covered\"`")
                        .with_span(self.span()))
                    }
                },
                _ => {
                    return Err(Diagnostic::error("expected string for missing_match")
                        .with_code_str(ErrCode::E004)
                        .with_help(
                            "`missing_match` expects a string literal — e.g. `with missing_match = \"message\"`",
                        )
                        .with_suggestion("use a string literal like `\"not all variants covered\"`")
                        .with_span(self.span()))
                }
            };
            self.expect(Token::Semicolon)?;
            Some(msg)
        } else {
            None
        };
        let enum_modifiers = self.parse_type_modifiers(bump)?;
        let end = self.span().end;
        Ok(Item::TypeDef {
            span: Span::new(self.file_id, start, end),
            attributes: bump.alloc_slice_clone(&attributes),
            doc,
            name,
            params: bump.alloc_slice_clone(&params),
            definition: TypeDefinition::Enum(
                bump.alloc_slice_clone(&variants),
                missing_match,
                bump.alloc_slice_clone(&enum_modifiers),
            ),
            contracts: &[],
        })
    }

    // -------------------------------------------------------------------------
    // GADT constraints: `Param == ConcreteType [and ...]`
    // -------------------------------------------------------------------------

    pub(super) fn parse_gadt_constraints(
        &mut self,
        bump: &'input bumpalo::Bump,
    ) -> Result<Vec<(Symbol, Type<'input>)>, Diagnostic<'src>> {
        let mut constraints = Vec::new();
        loop {
            let param_name = match self.advance() {
                Some(Token::Ident(name)) => name,
                Some(tok) => {
                    return Err(Diagnostic::error(format!(
                        "expected type parameter name in GADT constraint, found {:?}",
                        tok
                    ))
                    .with_code_str(ErrCode::E004)
                    .with_help("GADT constraints use `Param == ConcreteType` syntax")
                    .with_suggestion("use `T == Int<32>` where `T` is a type parameter of the enum")
                    .with_span(self.span()));
                }
                None => {
                    return Err(
                        Diagnostic::error("unexpected end of file in GADT constraint")
                            .with_code_str(ErrCode::E002)
                            .with_help("GADT constraints use `Param == ConcreteType` syntax")
                            .with_span(self.span()),
                    );
                }
            };
            self.expect(Token::EqEq)?;
            let concrete_ty = self.parse_type(bump)?;
            constraints.push((param_name, concrete_ty));
            match self.peek() {
                Some(Token::And) => {
                    self.advance();
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

    pub(super) fn parse_type_modifiers(
        &mut self,
        bump: &'input bumpalo::Bump,
    ) -> Result<Vec<TypeModifier<'input>>, Diagnostic<'src>> {
        let mut modifiers = Vec::new();
        while matches!(self.peek(), Some(Token::With)) {
            self.advance();
            match self.peek() {
                Some(Token::Ident(_)) | Some(Token::Default) | Some(Token::NoDefault) => {
                    let tok = self.advance().ok_or_else(|| Diagnostic::error("unexpected token")
                        .with_code_str(ErrCode::E003)
                        .with_help("expected a type modifier name (`overflow`, `validate`, `default`, `no_default`) after `with`")
                        .with_span(Span::new(self.file_id, 0, 0)))?;
                    match tok {
                        Token::Ident(ref s) if s.eq_str("overflow") => {
                            self.expect(Token::Assign)?;
                            let policy = match self.advance() {
                                Some(Token::Wrap) => OverflowPolicy::Wrap,
                                Some(Token::Saturate) => OverflowPolicy::Saturate,
                                Some(Token::Trap) => OverflowPolicy::Trap,
                                Some(Token::Ieee) => OverflowPolicy::Ieee,
                                _ => return Err(Diagnostic::error("expected overflow policy (wrap, saturate, trap, ieee)")
                                    .with_code_str(ErrCode::E007)
                                    .with_help("`overflow` policy must be one of: `wrap`, `saturate`, `trap`, or `ieee` (floats only)")
                                    .with_suggestion("use one of: `wrap`, `saturate`, `trap`, or `ieee`")
                                    .with_span(self.span())),
                            };
                            modifiers.push(TypeModifier::Overflow(policy));
                            if matches!(self.peek(), Some(Token::Semicolon)) {
                                self.advance();
                            }
                        }
                        Token::Ident(ref s) if s.eq_str("validate") => {
                            self.expect(Token::Assign)?;
                            let closure = self.parse_closure(self.span().start, bump)?;
                            modifiers.push(TypeModifier::Validate(closure));
                            if matches!(self.peek(), Some(Token::Semicolon)) {
                                self.advance();
                            }
                        }
                        Token::Default => {
                            self.expect(Token::Assign)?;
                            let expr = self.parse_expr(bump)?;
                            modifiers.push(TypeModifier::Default(expr));
                            if matches!(self.peek(), Some(Token::Semicolon)) {
                                self.advance();
                            }
                        }
                        Token::NoDefault => {
                            modifiers.push(TypeModifier::NoDefault);
                            if matches!(self.peek(), Some(Token::Semicolon)) {
                                self.advance();
                            }
                        }
                        _ => {
                            while !matches!(
                                self.peek(),
                                Some(Token::Semicolon) | Some(Token::RBrace) | None
                            ) {
                                self.advance();
                            }
                            if matches!(self.peek(), Some(Token::Semicolon)) {
                                self.advance();
                            }
                        }
                    }
                }
                _ => {
                    while !matches!(
                        self.peek(),
                        Some(Token::Semicolon) | Some(Token::RBrace) | None
                    ) {
                        self.advance();
                    }
                    if matches!(self.peek(), Some(Token::Semicolon)) {
                        self.advance();
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
        bump: &'input bumpalo::Bump,
    ) -> Result<Item<'input>, Diagnostic<'src>> {
        let start = self.span().start;
        let name = match self.advance() {
            Some(Token::Ident(name)) => name,
            Some(tok) => self
                .keyword_to_ident(&tok)
                .unwrap_or_else(|| Symbol::intern(&format!("{:?}", tok))),
            None => {
                return Err(
                    Diagnostic::error("unexpected end of file in layout definition")
                        .with_code_str(ErrCode::E002)
                        .with_span(self.span()),
                );
            }
        };
        self.expect(Token::LBrace)?;
        loop {
            if matches!(self.peek(), Some(Token::Comma)) {
                self.advance();
                continue;
            }
            if matches!(self.peek(), Some(Token::Semicolon)) {
                self.advance();
                break;
            }
            let attr_name = match self.advance() {
                Some(Token::Ident(name)) => name,
                Some(tok) => self
                    .keyword_to_ident(&tok)
                    .unwrap_or_else(|| Symbol::intern(&format!("{:?}", tok))),
                None => break,
            };
            let mut args = Vec::new();
            if matches!(self.peek(), Some(Token::LParen)) {
                self.advance();
                let arg_name = match self.advance() {
                    Some(Token::Ident(name)) => name,
                    Some(tok) => self
                        .keyword_to_ident(&tok)
                        .unwrap_or_else(|| Symbol::intern(&format!("{:?}", tok))),
                    None => Symbol::INVALID,
                };
                args.push(crate::ast::Expr::Ident(arg_name, self.span()));
                self.expect(Token::RParen)?;
            }
            attributes.push(Attribute {
                name: attr_name,
                args,
                named_args: Vec::new(),
                span: Span::new(self.file_id, start, self.span().end),
            });
        }
        self.expect(Token::RBrace)?;
        let end = self.span().end;
        Ok(Item::LayoutDef {
            name,
            attributes: bump.alloc_slice_clone(&attributes),
            span: Span::new(self.file_id, start, end),
        })
    }

    // -------------------------------------------------------------------------
    // Type capture parameters for `set auto<...>`
    // -------------------------------------------------------------------------

    pub(super) fn parse_type_capture_params(
        &mut self,
    ) -> Result<Vec<TypeParam<'input>>, Diagnostic<'src>> {
        self.advance();
        let mut params = Vec::new();
        loop {
            match self.peek() {
                Some(Token::Gt) | Some(Token::Shr) => {
                    self.expect_gt()?;
                    break;
                }
                Some(Token::Ident(name)) => {
                    let name = *name;
                    self.advance();
                    params.push(TypeParam {
                        name,
                        bounds: &[],
                        kind: TypeParamKind::Type,
                        span: self.span(),
                    });
                    if matches!(self.peek(), Some(Token::Comma)) {
                        self.advance();
                    } else if matches!(self.peek(), Some(Token::Gt) | Some(Token::Shr)) {
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
