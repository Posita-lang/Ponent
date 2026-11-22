#!/usr/bin/env python3
"""Run every machine check in locator_wrap_algorithms.py and the baseline
brute-force checker."""
import locator_wrap_algorithms as A
import verify_locator_wrap as V

def main():
    V.main()
    A.check_1a()
    A.check_1a_dp()
    A.check_1a_monge()
    A.check_strict()
    A.check_cdq()
    A.check_scaling()
    A.check_envelope()
    A.check_envelope_lex()
    A.check_stack_only()
    A.check_rank_envelope()
    A.check_A()
    A.check_B()
    A.check_C()
    A.check_window()
    A.check_lower_bound_construction()
    A.check_lower_bound_reduction()
    A.check_2(14, True)
    A.check_2_construction()
    A.check_3()
    A.check_3_exact()
    A.check_frontier_fast()
    A.check_frontier_structure()
    A.check_relational()
    A.check_regime_exhaustive()
    A.check_rank_convexity()
    A.check_rank()
    A.check_relational_secondary()
    A.check_reversed()
    A.find_arbitrary_counterexample()
    A.relational_counterexample()
    A.witness_1b()
    A.check_1b()
    print("ALL CHECKS COMPLETED")

if __name__ == "__main__":
    main()