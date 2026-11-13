#!/usr/bin/env python3
"""Operation-count model for `wrap_dp` vs four per-class Larmore-Schieber runs.

This is not a benchmark.  It makes the constants in the decision explicit:

  naive:  one monotone-window scan per (j, i) pair, ~c_scan ops each
  LS:     four per-class online Monge recurrences, ~c_LS ops per step each

`c_scan` and `c_LS` are the ranges quoted in `src/diagnostics/wrap.rs`.
"""

C_SCAN = (10, 12, 15)
C_LS = (30, 40, 50)


def naive_ops(n: int, k: int, c_scan: int) -> float:
    # One pass over the words that fit a line, for every prefix end.
    return c_scan * n * k


def ls_ops(n: int, alpha: float, c_ls: int) -> float:
    # Four fitness classes, each an independent online Monge recurrence.
    return 4.0 * c_ls * n * alpha


def main() -> None:
    print("crossover ratio = LS ops / naive ops (values < 1 mean LS cheaper)")
    print("alpha = inverse-Ackermann factor")
    print()

    for n in (50, 130, 500, 1000):
        print(f"n={n}")
        for k in (5, 10, 15, 20):
            row = []
            for c_scan in C_SCAN:
                worst = 0.0
                for c_ls in C_LS:
                    for alpha in (1.0, 2.0):
                        ratio = ls_ops(n, alpha, c_ls) / naive_ops(n, k, c_scan)
                        worst = max(worst, ratio)
                row.append(f"k={k:>2}: {worst:5.2f}")
            print("   " + "   ".join(row))
        print()

    print("Readings:")
    print("  * n <= 130, k <= 20: the ratio is ~1 or worse, and the pointer")
    print("    is pinned at the window edge, so LS prunes nothing.")
    print("  * The ratio falls below 1 only for a long unclosed window with")
    print("    large k; that is the measured-workload trigger, not the")
    print("    diagnostic corpus.")


if __name__ == "__main__":
    main()
