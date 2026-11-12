//! Witness tests for `research/underline-lanes/no-distributive-lattice.md`.
//!
//! Self-contained, no crate dependencies. Run standalone with
//!     rustc --test underline_lanes_witness_tests.rs -o witness_tests && ./witness_tests
//! or paste the `#[test]` functions into the crate's test module.
//!
//! Pins: Witness A (min/max leave C, antichain, choice of normalising clique),
//! Witness B (pointwise 2-chain vs refinement antichain), Witness C (6-element
//! antichain, transitive symmetry group, conjugation fixes only e), and the
//! corrected Stanley identities (decomposition + reciprocity) on K2 and K3.

type Colour = usize; // 1..=k
type Colouring = Vec<Colour>;

struct Graph {
    n: usize,
    edges: Vec<(usize, usize)>,
}

impl Graph {
    fn new(n: usize, edges: &[(usize, usize)]) -> Graph {
        Graph {
            n,
            edges: edges.to_vec(),
        }
    }
}

fn is_proper(g: &Graph, c: &[Colour]) -> bool {
    g.edges.iter().all(|&(a, b)| c[a] != c[b])
}

/// All proper `k`-colourings normalised by `c(clique[i]) = i + 1`.
fn normalised(g: &Graph, k: usize, clique: &[usize]) -> Vec<Colouring> {
    assert_eq!(clique.len(), k);
    let mut base: Vec<Option<Colour>> = vec![None; g.n];
    for (i, &v) in clique.iter().enumerate() {
        assert!(base[v].is_none());
        base[v] = Some(i + 1);
    }
    let free: Vec<usize> = (0..g.n).filter(|&v| base[v].is_none()).collect();
    let mut out = Vec::new();
    enumerate(g, k, &free, 0, &mut base, &mut out);
    out.sort();
    out.dedup();
    out
}

fn enumerate(
    g: &Graph,
    k: usize,
    free: &[usize],
    idx: usize,
    cur: &mut Vec<Option<Colour>>,
    out: &mut Vec<Colouring>,
) {
    if idx == free.len() {
        let c: Colouring = cur.iter().map(|x| x.unwrap()).collect();
        if is_proper(g, &c) {
            out.push(c);
        }
        return;
    }
    for col in 1..=k {
        cur[free[idx]] = Some(col);
        enumerate(g, k, free, idx + 1, cur, out);
    }
    cur[free[idx]] = None;
}

fn leq(a: &[Colour], b: &[Colour]) -> bool {
    a.iter().zip(b.iter()).all(|(x, y)| x <= y)
}

fn pmin(a: &[Colour], b: &[Colour]) -> Colouring {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| std::cmp::min(*x, *y))
        .collect()
}

fn pmax(a: &[Colour], b: &[Colour]) -> Colouring {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| std::cmp::max(*x, *y))
        .collect()
}

fn blocks(c: &[Colour], k: usize) -> Vec<Vec<usize>> {
    (1..=k)
        .map(|col| (0..c.len()).filter(|&v| c[v] == col).collect())
        .collect()
}

fn refines(p: &[Vec<usize>], q: &[Vec<usize>]) -> bool {
    p.iter()
        .all(|b| q.iter().any(|bl| b.iter().all(|v| bl.contains(v))))
}

fn perms3() -> Vec<[usize; 3]> {
    vec![
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ]
}

// ---------------------------------------------------------------- Witness A

#[test]
fn witness_a_pointwise_and_refinement_antichain() {
    // v1=0, v2=1, v3=2, u=3, v=4; two triangles sharing v3.
    let g = Graph::new(5, &[(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (3, 4)]);
    let c = normalised(&g, 3, &[0, 1, 2]);
    assert_eq!(c.len(), 2);
    let (a, b) = (&c[0], &c[1]);
    assert!(!leq(a, b) && !leq(b, a));
    assert!(!is_proper(&g, &pmin(a, b))); // (u,v) -> (1,1) breaks edge uv
    assert!(!is_proper(&g, &pmax(a, b))); // (u,v) -> (2,2) breaks edge uv
    let (pa, pb) = (blocks(a, 3), blocks(b, 3));
    assert!(!refines(&pa, &pb) && !refines(&pb, &pa));

    // The other maximum clique {u, v, v3} gives the same antichain.
    let c2 = normalised(&g, 3, &[3, 4, 2]);
    assert_eq!(c2.len(), 2);
    assert!(!leq(&c2[0], &c2[1]) && !leq(&c2[1], &c2[0]));
}

// ---------------------------------------------------------------- Witness B

#[test]
fn witness_b_pointwise_is_a_chain_but_refinement_is_not() {
    // a=0, b=1, c=2, pendant x=3 adjacent only to a.
    let g = Graph::new(4, &[(0, 1), (0, 2), (1, 2), (0, 3)]);
    let c = normalised(&g, 3, &[0, 1, 2]); // c(x) in {2, 3}
    assert_eq!(c.len(), 2);
    assert!(leq(&c[0], &c[1]) && !leq(&c[1], &c[0])); // pointwise: 2-chain
    assert!(c.contains(&pmin(&c[0], &c[1])));
    assert!(c.contains(&pmax(&c[0], &c[1])));
    let (p, q) = (blocks(&c[0], 3), blocks(&c[1], 3));
    assert!(!refines(&p, &q) && !refines(&q, &p)); // refinement: antichain
}

// ---------------------------------------------------------------- Witness C

fn two_triangles() -> Graph {
    Graph::new(6, &[(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)])
}

#[test]
fn witness_c_pointwise_antichain_and_block_type() {
    let g = two_triangles();
    let c = normalised(&g, 3, &[0, 1, 2]);
    assert_eq!(c.len(), 6);
    for i in 0..c.len() {
        for j in (i + 1)..c.len() {
            assert!(!leq(&c[i], &c[j]) && !leq(&c[j], &c[i]));
            assert!(!is_proper(&g, &pmin(&c[i], &c[j])));
        }
        assert!(blocks(&c[i], 3).iter().all(|b| b.len() == 2)); // (2,2,2)
    }
}

#[test]
fn witness_c_symmetry_is_transitive_and_conjugation_fixes_only_e() {
    let g = two_triangles();
    let c = normalised(&g, 3, &[0, 1, 2]);
    let p0: [Colour; 3] = [c[0][3], c[0][4], c[0][5]]; // (1,2,3) after sort
    let perms = perms3();

    // Normalisation-preserving group: p -> sigma o p o rho^{-1}.
    let mut orbit: Vec<[Colour; 3]> = Vec::new();
    for s in &perms {
        for r in &perms {
            let mut p = [0usize; 3];
            for i in 0..3 {
                let j = r.iter().position(|&x| x == i).unwrap(); // rho^{-1}(i)
                p[i] = s[p0[j] - 1] + 1; // sigma acting on colour values
            }
            if !orbit.contains(&p) {
                orbit.push(p);
            }
        }
    }
    assert_eq!(orbit.len(), 6, "group action must be transitive");
    for cc in &c {
        assert!(orbit.contains(&[cc[3], cc[4], cc[5]]));
    }

    // Conjugation p -> sigma o p o sigma^{-1}: the only fixed point is e.
    let mut fixed = 0;
    for &p in &orbit {
        let is_fixed = perms.iter().all(|s| {
            let mut q = [0usize; 3];
            for i in 0..3 {
                let j = s.iter().position(|&x| x == i).unwrap(); // sigma^{-1}(i)
                q[i] = s[p[j] - 1] + 1;
            }
            q == p
        });
        if is_fixed {
            fixed += 1;
        }
    }
    assert_eq!(fixed, 1);
}

// ------------------------------------------------------- Stanley identities

fn chi_kn(n: usize, t: isize) -> isize {
    match n {
        2 => t * (t - 1),         // K2
        3 => t * (t - 1) * (t - 2), // K3
        _ => panic!("only K2 and K3 are used"),
    }
}

fn acyclic(n: usize, arcs: &[(usize, usize)]) -> bool {
    let mut r = vec![vec![false; n]; n];
    for &(a, b) in arcs {
        r[a][b] = true;
    }
    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                r[i][j] = r[i][j] || (r[i][k] && r[k][j]);
            }
        }
    }
    (0..n).all(|i| !r[i][i])
}

fn orientations(n: usize, edges: &[(usize, usize)]) -> Vec<Vec<(usize, usize)>> {
    let mut out = Vec::new();
    for mask in 0..(1usize << edges.len()) {
        let arcs: Vec<(usize, usize)> = edges
            .iter()
            .enumerate()
            .map(|(i, &(a, b))| {
                if (mask >> i) & 1 == 0 {
                    (a, b)
                } else {
                    (b, a)
                }
            })
            .collect();
        if acyclic(n, &arcs) {
            out.push(arcs);
        }
    }
    out
}

/// Number of maps V -> [t] that are (strictly) order-preserving along arcs.
fn count_maps(n: usize, arcs: &[(usize, usize)], t: usize, strict: bool) -> usize {
    let total = t.pow(n as u32);
    let mut cnt = 0usize;
    for code in 0..total {
        let mut f = vec![0usize; n];
        let mut x = code;
        for slot in f.iter_mut() {
            *slot = 1 + x % t;
            x /= t;
        }
        let ok = arcs
            .iter()
            .all(|&(a, b)| if strict { f[a] < f[b] } else { f[a] <= f[b] });
        if ok {
            cnt += 1;
        }
    }
    cnt
}

#[test]
fn stanley_decomposition_and_reciprocity() {
    let cases: Vec<(usize, Vec<(usize, usize)>)> =
        vec![(2, vec![(0, 1)]), (3, vec![(0, 1), (0, 2), (1, 2)])];
    for (n, edges) in &cases {
        let n = *n;
        let ao = orientations(n, edges);
        let sign = (-1isize).pow(n as u32);
        assert_eq!(ao.len() as isize, sign * chi_kn(n, -1)); // #AO = (-1)^n chi(-1)

        for t in 1..=5usize {
            let strict: usize = ao.iter().map(|a| count_maps(n, a, t, true)).sum();
            let weak: usize = ao.iter().map(|a| count_maps(n, a, t, false)).sum();
            assert_eq!(strict as isize, chi_kn(n, t as isize)); // chi(t) = sum Omega^o(t)
            assert_eq!(
                weak as isize,
                sign * chi_kn(n, -(t as isize)) // reciprocity
            );
        }

        // The old draft's (ii): (-1)^n chi(-t) = sum_D t^{#sources(D)} is wrong.
        // (At t = 1 it happens to agree, so start at t = 2.)
        for t in 2..=5usize {
            let sources: usize = ao
                .iter()
                .map(|a| {
                    let s = (0..n)
                        .filter(|&v| !a.iter().any(|&(_, b)| b == v))
                        .count();
                    t.pow(s as u32)
                })
                .sum();
            assert_ne!(sources as isize, sign * chi_kn(n, -(t as isize)));
        }
    }
}
