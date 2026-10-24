use arrayvec::ArrayVec;
use crate::ast::{BinOp, Expr, Literal, Type, UnaryOp};
use crate::diagnostics::Diagnostic;
use crate::smallvec::{SmallVec, smallvec};

/// Compute the Damerau-Levenshtein (OSA) distance between two strings.
///
/// Counts an adjacent transposition (`lne`↔`len`) as a single edit, where
/// plain Levenshtein scores 2 — the `did_you_mean` suggestions otherwise
/// miss the most common keyboard typo.
///
/// All five buffers live inline on the stack for identifier-sized inputs
/// (≤ 8 chars); longer inputs spill to the heap via `SmallVec` instead of
/// being rejected, so the function stays total.  No allocation in the
/// common case.
pub(crate) fn osa_distance(a: &str, b: &str) -> usize {
    const INLINE: usize = 8;
    let a_chars: SmallVec<[char; INLINE]> = a.chars().collect();
    let b_chars: SmallVec<[char; INLINE]> = b.chars().collect();
    let a_len = a_chars.len();
    let b_len = b_chars.len();
    let mut prev_prev: SmallVec<[usize; INLINE]> = smallvec![0; b_len + 1];
    let mut prev: SmallVec<[usize; INLINE]> = (0..=b_len).collect();
    let mut curr: SmallVec<[usize; INLINE]> = smallvec![0; b_len + 1];
    for i in 1..=a_len {
        curr[0] = i;
        for j in 1..=b_len {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            let mut d = std::cmp::min(
                std::cmp::min(curr[j - 1] + 1, prev[j] + 1),
                prev[j - 1] + cost,
            );
            if i >= 2
                && j >= 2
                && a_chars[i - 1] == b_chars[j - 2]
                && a_chars[i - 2] == b_chars[j - 1]
            {
                d = std::cmp::min(d, prev_prev[j - 2] + 1);
            }
            curr[j] = d;
        }
        std::mem::swap(&mut prev_prev, &mut prev);
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b_len]
}

/// Find names in `candidates` that are similar to `name` (edit distance ≤ 2),
/// sorted by closest match. Returns up to 3 suggestions.
pub(crate) fn find_similar_names<'a>(name: &str, candidates: &'a [&'a str]) -> ArrayVec<&'a str, 3> {
    /// Distance threshold: the filter below keeps only `0 < d <= 3`.
    const MAX_DIST: usize = 3;
    let name_len = name.chars().count();
    let mut scored: Vec<(&str, usize)> = candidates
        .iter()
        // Length prefilter: each edit changes length by at most one, so
        // |Δlen| lower-bounds the distance and a candidate outside the
        // threshold by length alone can skip its DP.
        .filter(|c| name_len.abs_diff(c.chars().count()) <= MAX_DIST)
        .map(|c| (*c, osa_distance(name, c)))
        .filter(|(_, d)| *d <= MAX_DIST && *d > 0)
        .collect();
    scored.sort_by_key(|(_, d)| *d);
    scored.truncate(3);
    scored.into_iter().map(|(n, _)| n).collect()
}

/// Build a "did you mean ...?" suggestion string from similar names.
pub(crate) fn did_you_mean_suggestion(name: &str, candidates: &[&str]) -> Option<String> {
    let similar = find_similar_names(name, candidates);
    if similar.is_empty() {
        None
    } else {
        Some(format!("did you mean `{}`?", similar.join("`, `")))
    }
}

/// Check whether an expression is a valid assignment target (lvalue).
pub(crate) fn is_valid_lvalue<'input>(expr: &Expr<'input>) -> bool {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Adjacent transpositions cost 1 under OSA, 2 under plain Levenshtein.
    /// These are the keyboard-typo cases `did_you_mean` used to miss.
    #[test]
    fn transposition_is_one_edit() {
        assert_eq!(osa_distance("lne", "len"), 1);
        assert_eq!(osa_distance("retrun", "return"), 1);
        assert_eq!(osa_distance("E003", "E030"), 1);
    }

    /// Non-transposition distances are unchanged by the OSA swap.
    #[test]
    fn non_transposition_distances_unchanged() {
        assert_eq!(osa_distance("abc", "abc"), 0);
        assert_eq!(osa_distance("abc", "abd"), 1);
        assert_eq!(osa_distance("kitten", "sitting"), 3);
    }
}
