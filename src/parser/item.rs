//! Parsing of top‑level items: `def`, `trait`, `impl`, `extern`, `import`, `constraint`, etc.

use super::*;
use crate::ast::*;
use crate::diagnostics::Diagnostic;
use crate::lexer::Token;
use crate::symbol::Symbol;
use smallvec::smallvec;

impl<'input> Parser<'input> {
    // -------------------------------------------------------------------------
    // Item dispatch
    // -------------------------------------------------------------------------

    pub(super) fn parse_item(&mut self) -> Result<Stmt<'input>, Diagnostic> {
        // Collect leading @attributes and doc comments from the source.
        let mut attributes = Vec::new();
        let mut doc = None;
        loop {
            match self.peek() {
                Ok(Token::At) => attributes.push(self.parse_attribute()?),
                Ok(Token::DocComment(s)) => {
                    doc = Some(s.clone());
                    self.advance().ok();
                }
                Ok(Token::ModuleDocComment(s)) => {
                    doc = Some(s.clone());
                    self.advance().ok();
                }
                _ => break,
            }
        }
        match self.peek() {
            Ok(Token::Comptime) => self.parse_comptime_item(attributes, doc),
            Ok(Token::Async) => {
                self.advance().ok();
                self.expect(Token::Def)?;
                self.with_restrictions(ParseRestrictions::ALLOW_TYPE_PARAMS, |this| {
                    this.parse_function_def(attributes, doc, false, true)
                })
            }
            Ok(Token::Def) => {
                // Nested function definitions: parse them as function
                // definitions (rustc-style — items in blocks are collected
                // and referenced, not rejected).
                self.advance().ok();
                self.with_restrictions(ParseRestrictions::ALLOW_TYPE_PARAMS, |this| {
                    this.parse_function_def(attributes, doc, false, false)
                })
            }
            Ok(Token::Edition) => self.parse_edition(),
            Ok(Token::Import) | Ok(Token::From) => self.parse_import(),
            Ok(Token::Extern) => self.parse_extern_function(attributes),
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
                self.with_restrictions(ParseRestrictions::ALLOW_TYPE_PARAMS, |this| {
                    this.parse_type_def(attributes, doc)
                })
            }
            Ok(Token::Trait) => self.parse_trait_def(attributes, doc),
            Ok(Token::Impl) => self
                .with_restrictions(ParseRestrictions::ALLOW_TYPE_PARAMS, |this| {
                    this.parse_impl_block(attributes)
                }),
            Ok(Token::Constraint) => self.parse_constraint(),
            Ok(Token::Set) | Ok(Token::Let) => self.parse_variable_def(attributes),
            Ok(Token::Layout) => {
                self.advance().ok();
                self.parse_layout_def(attributes)
            }
            Ok(Token::Generate) => self.parse_generate_item(attributes),
            _ => {
                let tok = self.advance().ok();
                let mut diag = Diagnostic::error(format!("unexpected token at top level: {}", tok.as_ref().map(|t| t.to_user_string()).unwrap_or_else(|| "end of file".to_string())))
                    .with_code_str("E003")
                    .with_help("only items (`def`, `type`, `trait`, `import`, `edition`, `constraint`, `extern`, `impl`, `comptime`, `async`, `set`, `let`) are allowed at the top level")
                    .with_span(self.span());
                if let Some(Token::Ident(name)) = &tok {
                    if let Some(suggestion) =
                        did_you_mean_keyword(&name.as_str(), KeywordContext::TopLevel)
                    {
                        diag = diag.with_suggestion(suggestion);
                    } else {
                        diag = diag.with_suggestion("move this token inside a function body, or start a new top-level declaration");
                    }
                } else {
                    diag = diag.with_suggestion("enclose this in a function definition, or start a new top-level declaration");
                }
                if self.cascade_suppressed {
                    self.skip_to_next_top_level();
                    return Ok(Stmt::Error(Span::new(0, 0)));
                }
                self.cascade_suppressed = true;
                Err(diag)
            }
        }
    }

    // -------------------------------------------------------------------------
    // `comptime` items (def or block)
    // -------------------------------------------------------------------------

    fn parse_comptime_item(
        &mut self,
        attributes: Vec<Attribute<'input>>,
        doc: Option<String>,
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
                        .with_help("use `comptime @trusted { ... }` for a trusted comptime block")
                        .with_suggestion("remove the unknown attribute, or use `@trusted`"),
                );
            }
            true
        } else {
            false
        };
        match self.peek() {
            Ok(Token::Def) => {
                if trusted {
                    return Err(Diagnostic::error("`comptime @trusted def` is not valid")
                        .with_code_str("E004")
                        .with_help("for functions, use `@trusted comptime def f(...) { ... }` — `@trusted` should appear before `comptime`")
                        .with_suggestion("move `@trusted` before `comptime`: `@trusted comptime def f(...) { ... }`")
                        .with_span(self.span()));
                }
                self.advance().ok();
                self.with_restrictions(ParseRestrictions::ALLOW_TYPE_PARAMS, |this| {
                    this.parse_function_def(attributes, doc, true, false)
                })
            }
            Ok(Token::LBracket) => {
                let start = self.span().start;
                self.advance().ok();
                let mut captures = Vec::new();
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
            Ok(Token::LBrace) => {
                let start = self.span().start;
                self.expect(Token::LBrace)?;
                let body = self.parse_block()?;
                self.expect(Token::RBrace)?;
                let end = self.span().end;
                Ok(Stmt::ComptimeBlock {
                    captures: Vec::new(),
                    trusted,
                    attributes,
                    body,
                    span: Span::new(start, end),
                })
            }
            _ => {
                let tok = self.advance().ok();
                Err(Diagnostic::error(format!("expected 'def' or '{{' after comptime, found {:?}", tok))
                    .with_code_str("E004")
                    .with_help("`comptime` must be followed by `def` (to declare a comptime function) or `{` (to start a comptime block)")
                    .with_suggestion("try `comptime def name(...) { ... }` for a comptime function, or `comptime { ... }` for a comptime block")
                    .with_span(self.span()))
            }
        }
    }

    // -------------------------------------------------------------------------
    // Function definitions
    // -------------------------------------------------------------------------

    pub(super) fn parse_function_def(
        &mut self,
        attributes: Vec<Attribute<'input>>,
        doc: Option<String>,
        is_comptime: bool,
        is_async: bool,
    ) -> Result<Stmt<'input>, Diagnostic> {
        let start = self.span().start;
        let name = match self.advance() {
            Ok(Token::Ident(name)) => {
                if name.eq_str("layout_of") {
                    return Err(Diagnostic::error("`layout_of` is a reserved identifier and cannot be used as a function name")
                        .with_code_str("E004")
                        .with_span(self.span())
                        .with_help("`layout_of` is a built-in comptime intrinsic (see SYNTAX.md §Comptime Intrinsics)")
                        .with_suggestion("rename the function to something else, or use `layout_of!(Type)` syntax to invoke the builtin"));
                }
                name
            }
            Ok(tok) => return Err(Diagnostic::error(format!("expected function name, found {:?}", tok))
                .with_code_str("E004")
                .with_help("a function name must follow `def` — use a valid identifier")
                .with_suggestion("use a valid identifier like `my_function` — keywords cannot be used as function names")
                .with_span(self.span())),
            Err(()) => return Err(Diagnostic::error("unexpected end of file in function definition")
                .with_code_str("E002")
                .with_help("function definition is incomplete — expected a name after `def`")
                .with_suggestion("add a function name after `def`, e.g. `def main() { ... }`")
                .with_span(self.span())),
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
            Some(self.parse_type()?)
        } else {
            None
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
        let where_clause = if matches!(self.peek(), Ok(Token::Where)) {
            Some(self.parse_where_clause()?)
        } else {
            None
        };
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
            where_clause,
            finally,
            is_comptime,
            is_async,
        })
    }

    // -------------------------------------------------------------------------
    // Trait definitions
    // -------------------------------------------------------------------------

    pub(super) fn parse_trait_def(
        &mut self,
        attributes: Vec<Attribute<'input>>,
        doc: Option<String>,
    ) -> Result<Stmt<'input>, Diagnostic> {
        let start = self.span().start;
        self.advance().ok();
        let name =
            match self.advance() {
                Ok(Token::Ident(name)) => name,
                _ => return Err(Diagnostic::error("expected trait name")
                    .with_code_str("E004")
                    .with_help("`trait` must be followed by a name — e.g. `trait Display { ... }`")
                    .with_suggestion(
                        "add a trait name after `trait`, e.g. `trait MyTrait { def foo(&self); }`",
                    )
                    .with_span(self.span())),
            };
        if matches!(self.peek(), Ok(Token::Lt)) {
            let lt_span = self.span();
            let _type_params = self.parse_type_params()?;
            return Err(Diagnostic::error("generic traits are not yet supported")
                .with_code_str("E004")
                .with_help("trait definitions cannot have type parameters yet")
                .with_suggestion(
                    "remove the type parameters, or express constraints in the trait body",
                )
                .with_span(Span::new(lt_span.start, self.span().end)));
        }
        let mut methods = Vec::new();
        let mut associated_types = Vec::new();
        if matches!(self.peek(), Ok(Token::LBrace)) {
            self.expect(Token::LBrace)?;
            loop {
                if matches!(self.peek(), Ok(Token::RBrace)) {
                    self.advance().ok();
                    break;
                }
                match self.peek() {
                    Ok(Token::Type) => {
                        self.advance().ok();
                        let assoc_name = match self.advance() {
                            Ok(Token::Ident(n)) => n,
                            _ => return Err(Diagnostic::error("expected associated type name")
                                .with_code_str("E004")
                                .with_help("`type` in a trait body must be followed by a name — e.g. `type Output;`")
                                .with_suggestion("add an associated type name, e.g. `type Output;` or `type Item = Int<32>;`")
                                .with_span(self.span())),
                        };
                        let lifetime_params = self.parse_associated_type_lifetime_params()?;
                        let default = if matches!(self.peek(), Ok(Token::Assign)) {
                            self.advance().ok();
                            Some(self.parse_type()?)
                        } else {
                            None
                        };
                        self.expect(Token::Semicolon)?;
                        associated_types.push(AssociatedType {
                            name: assoc_name,
                            lifetime_params,
                            default,
                            span: Span::new(start, self.span().end),
                        });
                    }
                    Ok(Token::Def) => {
                        self.advance().ok();
                        let method_name = match self.advance() {
                            Ok(Token::Ident(n)) => n,
                            _ => return Err(Diagnostic::error("expected method name")
                                .with_code_str("E004")
                                .with_help("`def` in a trait body must be followed by a method name")
                                .with_suggestion("add a method name after `def`, e.g. `def method_name(&self) -> Int<32>;`")
                                .with_span(self.span())),
                        };
                        self.expect(Token::LParen)?;
                        let mut params = Vec::new();
                        loop {
                            match self.peek() {
                                Ok(Token::RParen) => { self.advance().ok(); break; }
                                Ok(Token::Ampersand) => {
                                    let param = self.parse_self_param()?;
                                    params.push(param);
                                }
                                Ok(Token::Ident(s)) if s.eq_str("self") || s.eq_str("SelfKw") => {
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
                        self.expect(Token::Semicolon)?;
                        methods.push(TraitMethod {
                            name: method_name,
                            params,
                            return_type,
                            span: Span::new(start, self.span().end),
                        });
                    }
                    _ => return Err(Diagnostic::error("expected 'type' or 'def' in trait body")
                        .with_code_str("E004")
                        .with_help("trait bodies can contain `type` (associated types) or `def` (method signatures)")
                        .with_suggestion("use `type AssocType;` for an associated type or `def method(&self);` for a method")
                        .with_span(self.span())),
                }
            }
        }
        let end = self.span().end;
        Ok(Stmt::TraitDef {
            span: Span::new(start, end),
            attributes,
            doc,
            name,
            methods,
            associated_types,
        })
    }

    // -------------------------------------------------------------------------
    // Impl blocks
    // -------------------------------------------------------------------------

    pub(super) fn parse_impl_block(
        &mut self,
        attributes: Vec<Attribute<'input>>,
    ) -> Result<Stmt<'input>, Diagnostic> {
        let start = self.span().start;
        self.advance().ok();
        let type_params = if matches!(self.peek(), Ok(Token::Lt)) {
            self.parse_type_params()?
        } else {
            Vec::new()
        };
        let (trait_path, for_type) = if matches!(self.peek(), Ok(Token::For)) {
            self.advance().ok();
            let for_ty = self.parse_type()?;
            (None, for_ty)
        } else {
            let first_type = self.parse_type()?;
            if matches!(self.peek(), Ok(Token::For)) {
                self.advance().ok();
                let for_ty = self.parse_type()?;
                (Some(Self::alloc_shared(self.arena, first_type)), for_ty)
            } else {
                (None, first_type)
            }
        };
        let where_clause = if matches!(self.peek(), Ok(Token::Where)) {
            Some(self.parse_where_clause()?)
        } else {
            None
        };
        self.expect(Token::LBrace)?;
        let mut methods = Vec::new();
        let mut associated_types = Vec::new();
        loop {
            if matches!(self.peek(), Ok(Token::RBrace)) {
                break;
            }
            if matches!(self.peek(), Ok(Token::Type)) {
                self.advance().ok();
                let assoc_name = match self.advance() {
                    Ok(Token::Ident(n)) => n,
                    _ => return Err(Diagnostic::error("expected associated type name")
                        .with_code_str("E004")
                        .with_help("`type` in an impl block must be followed by a name — e.g. `type Output = Int<32>;`")
                        .with_span(self.span())),
                };
                let lifetime_params = self.parse_associated_type_lifetime_params()?;
                self.expect(Token::Assign)?;
                let assoc_ty = self.parse_type()?;
                self.expect(Token::Semicolon)?;
                associated_types.push(AssociatedType {
                    name: assoc_name,
                    lifetime_params,
                    default: Some(assoc_ty),
                    span: Span::new(start, self.span().end),
                });
            } else {
                methods.push(self.parse_impl_method()?);
            }
        }
        self.expect(Token::RBrace)?;
        let end = self.span().end;
        Ok(Stmt::ImplBlock {
            span: Span::new(start, end),
            attributes,
            trait_path,
            for_type,
            methods,
            associated_types,
            where_clause,
            type_params,
        })
    }

    pub(super) fn parse_impl_method(&mut self) -> Result<ImplMethod<'input>, Diagnostic> {
        let mut attributes = Vec::new();
        while matches!(self.peek(), Ok(Token::At)) {
            attributes.push(self.parse_attribute()?);
        }
        if matches!(self.peek(), Ok(Token::Def)) {
            self.advance().ok();
        }
        let start = self.span().start;
        let name =
            match self.advance() {
                Ok(Token::Ident(name)) => name,
                Ok(tok) => {
                    return Err(Diagnostic::error(format!(
                        "expected method name, found {:?}",
                        tok
                    ))
                    .with_code_str("E004")
                    .with_help("a method name must follow `def` in an impl block")
                    .with_suggestion("use a valid identifier like `my_method` for the method name")
                    .with_span(self.span()));
                }
                Err(()) => return Err(Diagnostic::error(
                    "unexpected end of file in method definition",
                )
                .with_code_str("E002")
                .with_help("method definition is incomplete — expected a name after `def`")
                .with_suggestion(
                    "add a method name after `def`, e.g. `def process(&mut self) { ... }`",
                )
                .with_span(self.span())),
            };
        self.expect(Token::LParen)?;
        let mut params = Vec::new();
        loop {
            match self.peek() {
                Ok(Token::RParen) => {
                    self.advance().ok();
                    break;
                }
                Ok(Token::Ampersand) => {
                    let param = self.parse_self_param()?;
                    params.push(param);
                }
                Ok(Token::Ident(s)) if s.eq_str("self") || s.eq_str("SelfKw") => {
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
            attributes,
            params,
            return_type,
            body,
            span: Span::new(start, end),
        })
    }

    // -------------------------------------------------------------------------
    // `self` parameter
    // -------------------------------------------------------------------------

    pub(super) fn parse_self_param(&mut self) -> Result<Param<'input>, Diagnostic> {
        let start = self.span().start;
        let has_ampersand = matches!(self.peek(), Ok(Token::Ampersand));
        let mutable = if has_ampersand {
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
            Ok(Token::Ident(s)) if s.eq_str("self") => {
                let end = self.span().end;
                let ty: Type<'input> = if has_ampersand {
                    Type::Reference {
                        inner: Self::alloc_shared(
                            self.arena,
                            Type::Path(smallvec!["Self".into()], Span::new(start, end)),
                        ),
                        mutable,
                        lifetime: None,
                        span: Span::new(start, end),
                    }
                } else {
                    Type::Path(smallvec!["Self".into()], Span::new(start, end))
                };
                Ok(Param {
                    name: "self".into(),
                    ty: Some(ty),
                    default: None,
                    span: Span::new(start, end),
                })
            }
            Ok(Token::SelfKw) => {
                let end = self.span().end;
                let ty: Type<'input> = if has_ampersand {
                    Type::Reference {
                        inner: Self::alloc_shared(
                            self.arena,
                            Type::Path(smallvec!["Self".into()], Span::new(start, end)),
                        ),
                        mutable,
                        lifetime: None,
                        span: Span::new(start, end),
                    }
                } else {
                    Type::Path(smallvec!["Self".into()], Span::new(start, end))
                };
                Ok(Param {
                    name: "self".into(),
                    ty: Some(ty),
                    default: None,
                    span: Span::new(start, end),
                })
            }
            _ => Err(Diagnostic::error("expected 'self'")
                .with_code_str("E004")
                .with_help("method parameters must start with `self`, `&self`, or `&mut self`")
                .with_suggestion("try `self`, `&self`, or `&mut self` as the first parameter")
                .with_span(self.span())),
        }
    }

    // -------------------------------------------------------------------------
    // Function / closure parameter
    // -------------------------------------------------------------------------

    pub(super) fn parse_param(&mut self) -> Result<Param<'input>, Diagnostic> {
        let start = self.span().start;
        let name = match self.advance() {
            Ok(Token::Ident(name)) => name,
            Ok(tok) => {
                return Err(
                    Diagnostic::error(format!("expected parameter name, found {:?}", tok))
                        .with_code_str("E004")
                        .with_help("parameters must have a name — e.g. `def foo(x: Int<32>)`")
                        .with_suggestion(
                            "use a valid identifier like `x` or `value` for the parameter",
                        )
                        .with_span(self.span()),
                );
            }
            Err(()) => {
                return Err(
                    Diagnostic::error("unexpected end of file in parameter list")
                        .with_code_str("E002")
                        .with_help(
                            "parameter list is incomplete — expected a parameter name or `)`",
                        )
                        .with_suggestion("close the parameter list with `)` or add more parameters")
                        .with_span(self.span()),
                );
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

    // -------------------------------------------------------------------------
    // Contracts (requires / ensures / invariant / decreases / terminates)
    // -------------------------------------------------------------------------

    pub(super) fn parse_contract(&mut self) -> Result<Contract<'input>, Diagnostic> {
        let start = self.span().start;
        match self.advance().map_err(|_| Diagnostic::error("unexpected token")
            .with_code_str("E003")
            .with_help("unexpected syntax in contract — expected `requires`, `ensures`, `invariant`, `decreases`, or `terminates`")
            .with_span(Span::new(0, 0)))?
        {
            Token::Requires => {
                let expr = self.with_restrictions(ParseRestrictions::NO_STRUCT_LITERAL, |this| this.parse_expr())?;
                let end = self.span().end;
                Ok(Contract::Requires(expr, Span::new(start, end)))
            }
            Token::Ensures => {
                let mut target = EnsuresTarget::Unconditional;
                match self.peek() {
                    Ok(Token::OnTimeout) => {
                        self.advance().ok();
                        if !matches!(self.peek(), Ok(Token::FatArrow)) {
                            return Err(Diagnostic::error("expected '=>' after on_timeout")
                                .with_code_str("E004")
                                .with_help("`ensures on_timeout` must be followed by `=> <expression>`")
                                .with_suggestion("add `=> <expression>` after `on_timeout`")
                                .with_span(self.span()));
                        }
                        self.advance().ok();
                        target = EnsuresTarget::OnTimeout;
                    }
                    Ok(Token::OnCancel) => {
                        self.advance().ok();
                        if !matches!(self.peek(), Ok(Token::FatArrow)) {
                            return Err(Diagnostic::error("expected '=>' after on_cancel")
                                .with_code_str("E004")
                                .with_help("`ensures on_cancel` must be followed by `=> <expression>`")
                                .with_suggestion("add `=> <expression>` after `on_cancel`")
                                .with_span(self.span()));
                        }
                        self.advance().ok();
                        target = EnsuresTarget::OnCancel;
                    }
                    Ok(Token::On) => {
                        self.advance().ok();
                        match self.peek() {
                            Ok(Token::Ident(s)) if s.eq_str("Ok") => {
                                self.advance().ok();
                                self.expect(Token::LParen)?;
                                let pat = if !matches!(self.peek(), Ok(Token::RParen)) {
                                    Some(self.parse_pattern()?)
                                } else {
                                    None
                                };
                                self.expect(Token::RParen)?;
                                if !matches!(self.peek(), Ok(Token::FatArrow)) {
                                    return Err(Diagnostic::error("expected '=>' after on Ok(...)")
                                        .with_code_str("E004")
                                        .with_help("`ensures on Ok(...)` must be followed by `=> <expression>`")
                                        .with_suggestion("add `=> <expression>` after `on Ok(pat)`")
                                        .with_span(self.span()));
                                }
                                self.advance().ok();
                                target = EnsuresTarget::OnOk(pat);
                            }
                            Ok(Token::Ident(s)) if s.eq_str("Err") => {
                                self.advance().ok();
                                self.expect(Token::LParen)?;
                                let pat = if !matches!(self.peek(), Ok(Token::RParen)) {
                                    Some(self.parse_pattern()?)
                                } else {
                                    None
                                };
                                self.expect(Token::RParen)?;
                                if !matches!(self.peek(), Ok(Token::FatArrow)) {
                                    return Err(Diagnostic::error("expected '=>' after on Err(...)")
                                        .with_code_str("E004")
                                        .with_help("`ensures on Err(...)` must be followed by `=> <expression>`")
                                        .with_suggestion("add `=> <expression>` after `on Err(pat)`")
                                        .with_span(self.span()));
                                }
                                self.advance().ok();
                                target = EnsuresTarget::OnErr(pat);
                            }
                            _ => return Err(Diagnostic::error("expected 'Ok' or 'Err' after 'on'")
                                .with_code_str("E004")
                                .with_help("`ensures on` must be followed by `Ok(...)` or `Err(...)`")
                                .with_suggestion("try `ensures on Ok(result) => result != 0` or `ensures on Err(e) => e != 0`")
                                .with_span(self.span())),
                        }
                    }
                    _ => {}
                }
                let expr = self.with_restrictions(ParseRestrictions::NO_STRUCT_LITERAL, |this| this.parse_expr())?;
                let labels = Vec::new(); // @label support not yet in lexer
                let end = self.span().end;
                Ok(Contract::Ensures { expr, span: Span::new(start, end), target, labels })
            }
            Token::Invariant => {
                let expr = self.with_restrictions(ParseRestrictions::NO_STRUCT_LITERAL, |this| this.parse_expr())?;
                let end = self.span().end;
                Ok(Contract::Invariant(expr, Span::new(start, end)))
            }
            Token::Decreases => {
                let expr = self.with_restrictions(ParseRestrictions::NO_STRUCT_LITERAL, |this| this.parse_expr())?;
                let end = self.span().end;
                Ok(Contract::Decreases(expr, Span::new(start, end)))
            }
            Token::Terminates => {
                let expr = self.with_restrictions(ParseRestrictions::NO_STRUCT_LITERAL, |this| this.parse_expr())?;
                let end = self.span().end;
                Ok(Contract::Terminates(expr, Span::new(start, end)))
            }
            _ => unreachable!(),
        }
    }

    // -------------------------------------------------------------------------
    // Where clause
    // -------------------------------------------------------------------------

    pub(super) fn parse_where_clause(&mut self) -> Result<WhereClause<'input>, Diagnostic> {
        let start = self.span().start;
        self.advance().ok();
        let mut predicates = Vec::new();
        let mut equalities = Vec::new();
        let mut lifetime_outlives = Vec::new();
        loop {
            if matches!(self.peek(), Ok(Token::Apostrophe)) {
                self.advance().ok();
                let lt = match self.advance() {
                    Ok(Token::Ident(name)) => name,
                    _ => {
                        return Err(Diagnostic::error(
                            "expected a lifetime name after `'` in where clause",
                        )
                        .with_code_str("E004")
                        .with_span(self.span()));
                    }
                };
                self.expect(Token::Colon)?;
                let mut outlives = Vec::new();
                loop {
                    if !matches!(self.peek(), Ok(Token::Apostrophe)) {
                        return Err(Diagnostic::error(
                            "expected a lifetime (`'a`) after `:` in a lifetime outlives predicate",
                        )
                        .with_code_str("E004")
                        .with_span(self.span()));
                    }
                    self.advance().ok();
                    let bound = match self.advance() {
                        Ok(Token::Ident(name)) => name,
                        _ => {
                            return Err(Diagnostic::error(
                                "expected a lifetime name after `'` in where clause",
                            )
                            .with_code_str("E004")
                            .with_span(self.span()));
                        }
                    };
                    outlives.push(bound);
                    if !matches!(self.peek(), Ok(Token::Plus)) {
                        break;
                    }
                    self.advance().ok();
                }
                lifetime_outlives.push((lt, outlives));
            } else {
                let ty = self.parse_type()?;
                if matches!(self.peek(), Ok(Token::EqEq)) {
                    self.advance().ok();
                    let right = self.parse_type()?;
                    equalities.push(WhereEquality {
                        left: ty,
                        right,
                        span: Span::new(start, self.span().end),
                    });
                } else {
                    self.expect(Token::Colon)?;
                    let mut bounds = Vec::new();
                    loop {
                        bounds.push(self.parse_type()?);
                        if !matches!(self.peek(), Ok(Token::Plus)) {
                            break;
                        }
                        self.advance().ok();
                    }
                    let end = self.span().end;
                    predicates.push(WherePredicate {
                        ty,
                        bounds,
                        span: Span::new(start, end),
                    });
                }
            }
            if !matches!(self.peek(), Ok(Token::Comma)) {
                break;
            }
            self.advance().ok();
        }
        Ok(WhereClause {
            predicates: smallvec::SmallVec::from(predicates),
            equalities,
            lifetime_outlives,
        })
    }

    // -------------------------------------------------------------------------
    // Extern function
    // -------------------------------------------------------------------------

    pub(super) fn parse_extern_function(
        &mut self,
        attributes: Vec<Attribute<'input>>,
    ) -> Result<Stmt<'input>, Diagnostic> {
        let start = self.span().start;
        self.advance().ok();
        let abi = match self.advance() {
            Ok(Token::StringLiteral(Ok(s))) => s,
            _ => {
                return Err(Diagnostic::error("expected ABI string after 'extern'")
                    .with_code_str("E004")
                    .with_help("`extern` must be followed by an ABI string — e.g. `extern \"C\"`")
                    .with_suggestion("add an ABI string like `\"C\"` after `extern`")
                    .with_span(self.span()));
            }
        };
        self.expect(Token::Def)?;
        let name =
            match self.advance() {
                Ok(Token::Ident(name)) => name,
                _ => return Err(Diagnostic::error("expected function name")
                    .with_code_str("E004")
                    .with_help("after `extern \"<ABI>\" def`, a function name is expected")
                    .with_suggestion(
                        "add a function name after `def`, e.g. `extern \"C\" def my_function()`",
                    )
                    .with_span(self.span())),
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

    // -------------------------------------------------------------------------
    // Edition
    // -------------------------------------------------------------------------

    pub(super) fn parse_edition(&mut self) -> Result<Stmt<'input>, Diagnostic> {
        let start = self.span().start;
        self.advance().ok();
        self.expect(Token::Assign)?;
        let edition = match self.advance() {
            Ok(Token::StringLiteral(Ok(s))) => s,
            Ok(tok) => return Err(Diagnostic::error(format!("expected edition string, found {:?}", tok))
                .with_code_str("E004")
                .with_help("`edition = \"<version>\"` expects a string literal — e.g. `edition = \"2024\"`")
                .with_suggestion("use a string literal like `\"2024\"` for the edition")
                .with_span(self.span())),
            Err(()) => return Err(Diagnostic::error("unexpected end of file in edition declaration")
                .with_code_str("E002")
                .with_help("`edition = \"<version>\"` declaration is incomplete — expected a string literal after `=`")
                .with_suggestion("add a string literal after `=`, e.g. `edition = \"2024\";`")
                .with_span(self.span())),
        };
        self.expect(Token::Semicolon)?;
        let end = self.span().end;
        Ok(Stmt::Edition(edition, Span::new(start, end)))
    }

    // -------------------------------------------------------------------------
    // Import
    // -------------------------------------------------------------------------

    pub(super) fn parse_import(&mut self) -> Result<Stmt<'input>, Diagnostic> {
        let start = self.span().start;
        let is_from = matches!(self.peek(), Ok(Token::From));
        if is_from {
            self.advance().ok();
        }
        let mut path = Vec::new();
        match self.advance() {
            Ok(Token::Star) => return Err(Diagnostic::error("wildcard import is prohibited: `import *` is illegal")
                .with_code_str("E005")
                .with_help("explicit imports improve clarity and maintainability — list the items you need, or use `import path` for the module itself")
                .with_suggestion("use named imports: `import path::Item` or `from path import { Item1, Item2 }`")
                .with_span(self.span())),
            Ok(Token::Ident(part)) => path.push(part),
            _ => return Err(Diagnostic::error("expected module path")
                .with_code_str("E004")
                .with_help("after `import` or `from`, provide a module path — e.g. `import std::collections`")
                .with_suggestion("add a module path like `std::collections` or `my_module`")
                .with_span(self.span())),
        }
        while matches!(self.peek(), Ok(Token::ColonColon)) {
            self.advance().ok();
            if matches!(self.peek(), Ok(Token::LBrace)) {
                break;
            }
            match self.advance() {
                Ok(Token::Star) => {
                    return Err(Diagnostic::error("wildcard import is prohibited")
                        .with_code_str("E005")
                        .with_help("explicit imports improve clarity and maintainability")
                        .with_suggestion("use named imports: `import path::Item`")
                        .with_span(self.span()));
                }
                Ok(Token::Ident(part)) => path.push(part),
                _ => return Err(Diagnostic::error("expected identifier after '::'")
                    .with_code_str("E004")
                    .with_help(
                        "`::` must be followed by an identifier — e.g. `std::collections::HashMap`",
                    )
                    .with_suggestion("add an identifier after `::`, e.g. `MyModule::MyType`")
                    .with_span(self.span())),
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
                    Ok(Token::Star) => return Err(Diagnostic::error("wildcard import is prohibited")
                        .with_code_str("E005")
                        .with_help("explicit imports improve clarity and maintainability")
                        .with_suggestion("use named imports: `import path::Item`")
                        .with_span(self.span())),
                    Ok(Token::Ident(item)) => item,
                    _ => return Err(Diagnostic::error("expected import item name")
                        .with_code_str("E004")
                        .with_help("import items must be identifiers — e.g. `import std::{HashMap, HashSet}`")
                        .with_suggestion("list specific item names: `import path::{Item1, Item2}`")
                        .with_span(self.span())),
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
                    _ => return Err(Diagnostic::error("expected alias name")
                        .with_code_str("E004")
                        .with_help(
                            "`as` must be followed by an alias name — e.g. `import path as alias`",
                        )
                        .with_suggestion(
                            "add an alias name after `as`, e.g. `import path as MyAlias`",
                        )
                        .with_span(self.span())),
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
                    Ok(Token::Star) => {
                        return Err(Diagnostic::error("wildcard import is prohibited")
                            .with_code_str("E005")
                            .with_span(self.span()));
                    }
                    Ok(Token::Ident(item)) => items.push(item),
                    _ => {
                        return Err(Diagnostic::error("expected import item name")
                            .with_code_str("E004")
                            .with_span(self.span()));
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
                        return Err(Diagnostic::error("expected ',' or '}' in import list")
                            .with_code_str("E004")
                            .with_span(self.span()));
                    }
                }
            }
            Some(items)
        } else {
            None
        };
        let alias =
            if matches!(self.peek(), Ok(Token::As)) {
                self.advance().ok();
                match self.advance() {
                    Ok(Token::Ident(a)) => Some(a),
                    _ => return Err(Diagnostic::error("expected alias name")
                        .with_code_str("E004")
                        .with_help(
                            "`as` must be followed by an alias name — e.g. `import path as alias`",
                        )
                        .with_suggestion(
                            "add an alias name after `as`, e.g. `import path as MyAlias`",
                        )
                        .with_span(self.span())),
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

    // -------------------------------------------------------------------------
    // Constraint definition
    // -------------------------------------------------------------------------

    pub(super) fn parse_constraint(&mut self) -> Result<Stmt<'input>, Diagnostic> {
        let start = self.span().start;
        self.advance().ok();
        let name = match self.advance() {
            Ok(Token::Ident(name)) => name,
            _ => return Err(Diagnostic::error("expected constraint name")
                .with_code_str("E004")
                .with_help("`constraint` must be followed by a name — e.g. `constraint MyConstraint { ... }`")
                .with_suggestion("add a name after `constraint`, e.g. `constraint MyConstraint { T: Display + Debug }`")
                .with_span(self.span())),
        };
        let params = if matches!(self.peek(), Ok(Token::Lt)) {
            self.parse_type_params()?
        } else {
            Vec::new()
        };
        self.expect(Token::LBrace)?;
        let mut predicates = Vec::new();
        loop {
            if matches!(self.peek(), Ok(Token::RBrace)) {
                self.advance().ok();
                break;
            }
            let pred_start = self.span().start;
            let subject = self.parse_type()?;
            if !matches!(self.peek(), Ok(Token::Colon)) {
                return Err(Diagnostic::error(
                    "expected `:` after subject type in constraint predicate",
                )
                .with_code_str("E004")
                .with_help("constraint predicates must have the form `Subject: Bound1 + Bound2`")
                .with_suggestion("add a colon after the subject type, e.g. `T: Display + Debug`")
                .with_span(self.span()));
            }
            self.advance().ok();
            let mut bs = vec![self.parse_type()?];
            while matches!(self.peek(), Ok(Token::Plus)) {
                self.advance().ok();
                bs.push(self.parse_type()?);
            }
            predicates.push(WherePredicate {
                ty: subject,
                bounds: bs,
                span: Span::new(pred_start, self.span().end),
            });
            if !matches!(self.peek(), Ok(Token::Comma)) {
                break;
            }
            self.advance().ok();
        }
        self.expect(Token::RBrace)?;
        let end = self.span().end;
        Ok(Stmt::Constraint {
            name,
            params,
            predicates: smallvec::SmallVec::from(predicates),
            span: Span::new(start, end),
        })
    }

    // -------------------------------------------------------------------------
    // Generate block
    // -------------------------------------------------------------------------

    fn parse_generate_item(
        &mut self,
        attributes: Vec<Attribute<'input>>,
    ) -> Result<Stmt<'input>, Diagnostic> {
        let start = self.span().start;
        self.advance().ok();
        match self.advance() {
            Ok(Token::For) => {}
            Ok(tok) => {
                return Err(Diagnostic::error(format!(
                    "expected `for` after `generate`, found {:?}",
                    tok
                ))
                .with_code_str("E004")
                .with_help("`generate` must be followed by `for` and a type name or module path")
                .with_span(self.span()));
            }
            Err(()) => {
                return Err(Diagnostic::error(
                    "expected `for` after `generate`, found end of file",
                )
                .with_code_str("E004")
                .with_span(self.span()));
            }
        }
        let for_type = Self::alloc_shared(self.arena, self.parse_type()?);
        self.expect(Token::LBrace)?;
        let body = self.parse_block()?;
        self.expect(Token::RBrace)?;
        let end = self.span().end;
        Ok(Stmt::Generate {
            attributes,
            for_type,
            body,
            span: Span::new(start, end),
        })
    }
}
