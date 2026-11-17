#!/usr/bin/env python3
"""Verify the exact-chi padding reduction (§1 and §5 of ``../README.md``).

The reduction takes a hard ``p``-colour instance ``H`` (``p >= 2*omega(H) - 1``)
and forces ``chi = omega = p`` without removing the hardness.  Two gadgets are
checked here, both by brute force over every colouring:

  (A) disjoint padding — add a ``p``-clique on a disjoint time window.
      Then ``OPT_{k=chi}(G) = OPT_{p-colour}(H) + W * p(p+1)/2`` with ``W`` the
      common weight of the padding clique.

  (B) connected padding — attach the ``p``-clique as a pendant clique through a
      single staggered bridge interval, so ``G`` is connected, ``omega = p``,
      and every core clique is still strictly smaller than ``p``.  Same
      identity.

The point of (B) is that connectivity does not restore tractability: the
polynomial "fully saturated" algorithm of README §2 needs *every* maximal
clique at size ``p``, and this gadget preserves the unsaturated slack where the
hardness lives.

Closed-interval convention, as everywhere in this directory: intervals sharing
an endpoint conflict.  Standard library only.
"""

from __future__ import annotations

from itertools import product


# --------------------------------------------------------------- structure
def conflicts(a: tuple[int, int], b: tuple[int, int]) -> bool:
    (s1, e1), (s2, e2) = a, b
    return s1 <= e2 and s2 <= e1


def maximal_cliques(iv) -> list[tuple[int, ...]]:
    """All inclusion-maximal cliques, by subset enumeration (small n only)."""
    n = len(iv)
    out = []
    for mask in range(1, 1 << n):
        verts = [v for v in range(n) if mask >> v & 1]
        if any(
            not conflicts(iv[u], iv[v])
            for i, u in enumerate(verts)
            for v in verts[i + 1 :]
        ):
            continue
        outside = [v for v in range(n) if not mask >> v & 1]
        # maximal iff every outside vertex is non-adjacent to some clique member
        if all(any(not conflicts(iv[v], iv[w]) for w in verts) for v in outside):
            out.append(tuple(verts))
    return out


def omega_of(cliques) -> int:
    return max(len(k) for k in cliques)


def connected(iv) -> bool:
    n = len(iv)
    if n == 0:
        return True
    seen = {0}
    stack = [0]
    while stack:
        u = stack.pop()
        for v in range(n):
            if v not in seen and conflicts(iv[u], iv[v]):
                seen.add(v)
                stack.append(v)
    return len(seen) == n


def is_proper(c, cliques) -> bool:
    return all(len({c[v] for v in K}) == len(K) for K in cliques)


def chi_of(iv, cliques) -> int:
    """Smallest t for which the graph is t-colourable (brute force)."""
    n = len(iv)
    for t in range(1, n + 1):
        for c in product(range(1, t + 1), repeat=n):
            if is_proper(c, cliques):
                return t
    raise AssertionError("no colouring found")


def opt(iv, cliques, weights, p) -> int:
    """min sum w_v c(v) over proper colourings into 1..p (brute force)."""
    n = len(iv)
    best = None
    for c in product(range(1, p + 1), repeat=n):
        if not is_proper(c, cliques):
            continue
        val = sum(w * q for w, q in zip(weights, c))
        if best is None or val < best:
            best = val
    assert best is not None, "instance is not p-colourable"
    return best


# ----------------------------------------------------------------- gadgets
# Core H: a path of four intervals, omega = chi = 2, connected.
CORE = [(0, 3), (2, 5), (4, 7), (6, 9)]
P = 3


def disjoint_gadget(core, p):
    """Core on the left, a p-clique on a disjoint window."""
    iv = list(core) + [(20, 22)] * p
    return iv


def connected_gadget(core, p):
    """Core on the left; p-clique with staggered left endpoints at the right.

    ``u`` (the core interval with the largest right endpoint) is extended to
    ``r`` with ``l_1 < r < l_2``, so it overlaps only ``x_1``.
    """
    iv = list(core)
    u = max(range(len(core)), key=lambda v: core[v][1])
    s_u, _ = core[u]
    lefts = [14 + 3 * i for i in range(p)]  # l_1 < l_2 < ... < l_p
    end = 24
    r = lefts[0] + 1  # strictly between l_1 and l_2
    assert lefts[0] < r < lefts[1]
    iv[u] = (s_u, r)  # extend u
    iv.extend((left, end) for left in lefts)
    return iv, u, len(core)


def report(tag, iv, core_n, weights_core, p, u=None):
    cliques = maximal_cliques(iv)
    om, chi = omega_of(cliques), chi_of(iv, cliques)
    core_cliques = maximal_cliques(iv[:core_n])
    hl = max(len(k) for k in core_cliques)
    weights = list(weights_core) + [1] * p  # padding clique: unit weights
    core_opt = opt(iv[:core_n], core_cliques, weights_core, p)
    whole_opt = opt(iv, cliques, weights, p)
    const = p * (p + 1) // 2  # sum_{j=1..p} j for the clique, weight 1
    sizes = sorted({len(k) for k in cliques})

    print(f"[{tag}]")
    print(f"  n = {len(iv)}, chi = omega = {chi} (omega(H) = {hl}), "
          f"connected = {connected(iv)}")
    print(f"  maximal clique sizes: {sizes}")
    print(f"  OPT(H, p={p}) = {core_opt}, OPT(G) = {whole_opt}, "
          f"padding constant = {const}")
    assert om == chi == p
    assert whole_opt == core_opt + const, (whole_opt, core_opt + const)
    assert hl < p, "core must stay unsaturated"
    assert p in sizes and any(s < p for s in sizes), "gadget must be unsaturated"
    if u is not None:
        assert connected(iv)
        clique = list(range(core_n, len(iv)))
        assert conflicts(iv[u], iv[clique[0]])
        assert all(not conflicts(iv[u], iv[x]) for x in clique[1:])
        print(f"  bridge: u = v{u} touches only x_1, not x_2..x_{p}")
    print()


def main() -> None:
    print("Exact-chi padding reduction: brute-force check of "
          "OPT_{k=chi}(G) = OPT_{p-colour}(H) + const\n")

    # Nonnegative weights, then a mixed-sign vector that pushes the bridge
    # interval u towards the *largest* colour (exercising the permutation
    # argument: x_1 then takes a lower colour and the clique constant is
    # unchanged).
    weight_sets = [
        ("nonnegative", [1, 2, 3, 1]),
        ("mixed-sign (u wants colour 3)", [1, 2, 3, -5]),
        ("zero weights", [0, 0, 0, 0]),
    ]

    for tag, w in weight_sets:
        report(f"disjoint, {tag}", disjoint_gadget(CORE, P), len(CORE), w, P)

    for tag, w in weight_sets:
        iv, u, core_n = connected_gadget(CORE, P)
        report(f"connected, {tag}", iv, core_n, w, P, u=u)

    # A negative weight on the bridge interval must not break the identity,
    # and neither may a weight that makes the core optimum use colour p.
    print("ALL CHECKS PASSED")
    print("  (both gadgets keep every core maximal clique below p, so the "
          "fully saturated\n   flow algorithm of README §2 does not apply)")
    print("  (connectivity is therefore not the tractability boundary; "
          "saturation is)")


if __name__ == "__main__":
    main()
