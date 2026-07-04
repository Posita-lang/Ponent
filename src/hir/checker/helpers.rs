use crate::ast::{BinOp, Expr, Literal, Type, UnaryOp};
use crate::diagnostics::Diagnostic;

/// Compute the Levenshtein distance between two strings.
pub(crate) fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<_> = a.chars().collect();
    let b_chars: Vec<_> = b.chars().collect();
    let a_len = a_chars.len();
    let b_len = b_chars.len();
    let mut prev: Vec<usize> = (0..=b_len).collect();
    let mut curr = vec![0; b_len + 1];
    for i in 1..=a_len {
        curr[0] = i;
        for j in 1..=b_len {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            curr[j] = std::cmp::min(
                std::cmp::min(curr[j - 1] + 1, prev[j] + 1),
                prev[j - 1] + cost,
            );
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b_len]
}

/// Find names in `candidates` that are similar to `name` (edit distance ≤ 2),
/// sorted by closest match. Returns up to 3 suggestions.
pub(crate) fn find_similar_names<'a>(name: &str, candidates: &'a [String]) -> Vec<&'a str> {
    let mut scored: Vec<(&str, usize)> = candidates
        .iter()
        .map(|c| (c.as_str(), levenshtein_distance(name, c)))
        .filter(|(_, d)| *d <= 3 && *d > 0)
        .collect();
    scored.sort_by_key(|(_, d)| *d);
    scored.truncate(3);
    scored.into_iter().map(|(n, _)| n).collect()
}

/// Build a "did you mean ...?" suggestion string from similar names.
pub(crate) fn did_you_mean_suggestion(name: &str, candidates: &[String]) -> Option<String> {
    let similar = find_similar_names(name, candidates);
    if similar.is_empty() {
        None
    } else {
        Some(format!("did you mean `{}`?", similar.join("`, `")))
    }
}

/// Check whether an expression is a valid assignment target (lvalue).
pub(crate) fn is_valid_lvalue(expr: &Expr) -> bool {
    match expr {
        Expr::Ident(_, _) => true,
        Expr::FieldAccess { .. } => true,
        Expr::Index { .. } => true,
        Expr::UnaryOp {
            op: UnaryOp::Deref, ..
        } => true,
        Expr::UnaryOp {
            op: UnaryOp::Ref,
            expr,
            ..
        }
        | Expr::UnaryOp {
            op: UnaryOp::RefMut,
            expr,
            ..
        } => is_valid_lvalue(expr),
        _ => false,
    }
}
