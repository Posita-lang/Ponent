//! Thin wrapper around the `sexp` crate adding multi-form parsing.
//!
//! The `sexp` crate only exposes [`sexp::parse`], which errors on trailing
//! data after the first top-level form.  This module adds [`parse_many`],
//! which collects every top-level form in the input — needed for parsing
//! SMT-LIB model output where multiple `define-fun` forms appear in
//! sequence.

use sexp::Sexp;

/// Parse multiple top-level s-expressions from a string.
///
/// Unlike [`sexp::parse`], which errors on trailing data, this collects
/// every top-level form in the input. Whitespace and comments between
/// forms are skipped.
///
/// Returns an error if any form fails to parse.
pub fn parse_many(s: &str) -> Result<Vec<Sexp>, Box<sexp::Error>> {
    let mut result = Vec::new();
    let mut pos = 0;

    loop {
        // Skip whitespace and comments
        while pos < s.len() {
            let rest = &s[pos..];
            if rest.starts_with(';') {
                match rest.find('\n') {
                    Some(nl) => pos += nl + 1,
                    None => return Ok(result),
                }
            } else if let Some(c) = rest.chars().next() {
                if c.is_whitespace() {
                    pos += c.len_utf8();
                } else {
                    break;
                }
            }
        }

        if pos >= s.len() {
            break;
        }

        // Find the end of this top-level form by tracking paren depth
        let end = form_end(s, pos)?;
        result.push(sexp::parse(&s[pos..end])?);
        pos = end;
    }

    Ok(result)
}

/// Find the byte position just past the end of the top-level form starting
/// at `start`. Tracks parenthesis depth (respecting strings and comments)
/// to find where the form ends.
fn form_end(s: &str, start: usize) -> Result<usize, Box<sexp::Error>> {
    let mut pos = start;
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;

    while pos < s.len() {
        let c = s[pos..].chars().next().unwrap();
        let clen = c.len_utf8();

        if in_string {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
        } else {
            match c {
                '"' => in_string = true,
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        return Ok(pos + clen);
                    }
                }
                _ if depth == 0 && (c.is_whitespace() || c == ';') => {
                    return Ok(pos);
                }
                _ => {}
            }
        }
        pos += clen;
    }

    if depth == 0 {
        Ok(pos)
    } else {
        Err(Box::new(sexp::Error {
            message: "unexpected eof",
            line: 0,
            column: 0,
            index: pos,
        }))
    }
}

#[cfg(all(test, not(feature = "shuttle")))]
mod tests {
    use super::*;

    #[test]
    fn test_parse_many_basic() {
        let input = "(foo 1) (bar 2)";
        let forms = parse_many(input).unwrap();
        assert_eq!(forms.len(), 2);
    }

    #[test]
    fn test_parse_many_with_comments() {
        let input = "; comment\n(foo 1)\n; another\n(bar 2)\n";
        let forms = parse_many(input).unwrap();
        assert_eq!(forms.len(), 2);
    }

    #[test]
    fn test_parse_many_empty() {
        let forms = parse_many("").unwrap();
        assert!(forms.is_empty());
    }

    #[test]
    fn test_parse_many_only_comments() {
        let forms = parse_many("; just a comment\n").unwrap();
        assert!(forms.is_empty());
    }

    #[test]
    fn test_parse_many_nested() {
        let input = "(foo (bar (baz 1))) (qux 2)";
        let forms = parse_many(input).unwrap();
        assert_eq!(forms.len(), 2);
    }

    #[test]
    fn test_parse_many_strings_with_parens() {
        let input = r#"(foo "(bar)") (baz 1)"#;
        let forms = parse_many(input).unwrap();
        assert_eq!(forms.len(), 2);
    }
}
