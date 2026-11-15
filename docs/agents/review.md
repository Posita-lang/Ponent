# Code Review Protocol

This document defines the review standards and workflow for AI agents auditing Ponent.

The objective is to identify concrete correctness defects, performance problems, edge cases, architectural weaknesses, and algorithmic optimization opportunities. A review must be technically rigorous, evidence-driven, and sufficiently detailed to support independent implementation.

## 1. Review Mission

Review the assigned code and its surrounding context for:

- Correctness and semantic consistency.
- Memory safety and lifetime correctness.
- Concurrency and synchronization correctness.
- Error handling and failure-path behavior.
- Boundary conditions and unusual inputs.
- Runtime performance and memory efficiency.
- Algorithmic complexity and optimization opportunities.
- Architectural inconsistencies and violated invariants.
- Compile-time and generated-code costs where relevant.

Do not restrict the review to obvious bugs or local code quality. Actively investigate algorithmic optimization opportunities, including high-effort and troublesome ones.

## 2. Investigation Requirements

Before reporting a finding:

1. Read the relevant implementation in sufficient depth.
2. Inspect callers, callees, and related data structures.
3. Trace relevant execution paths.
4. Examine existing tests and identify missing coverage.
5. Identify relevant architectural and type-level invariants.
6. Determine whether the suspected issue is reachable.
7. Verify the finding as far as the available environment permits.

For performance findings, investigate the actual cost model rather than relying exclusively on stylistic preferences. Consider time and space complexity, allocation behavior, memory retention, cache locality, data layout, cloning, copying, hashing, synchronization, lock contention, parallelism, scheduling, incremental computation, cache reuse, compile-time overhead, and code size.

For concurrency-related code, examine race conditions, lock ordering, panic behavior, cancellation, cache consistency, and object lifecycle where applicable.

## 3. Review Discipline

Our engineering principle is:

**The perfection of the project takes precedence over the comfort of its engineers.**

Do not avoid important optimization opportunities merely because they are difficult to implement.

However:

- Do not recommend complexity without a defensible benefit.
- Do not report stylistic preferences as correctness defects.
- Do not report speculative issues as confirmed bugs.
- Do not recommend replacing an existing design without understanding its trade-offs.
- Do not claim an optimization is faster without measurements or a sound complexity argument.
- Do not duplicate existing findings.
- Do not modify source code during a review-only task.

Prioritize findings by technical significance, not by how easy they are to explain or fix.

### Epistemic Discipline

A review is an investigation, not an exercise in producing plausible-sounding findings.

- Separate **observed facts**, **logical deductions**, **assumptions**, **hypotheses**, and **unknowns**.
- Every consequential finding must identify concrete evidence: relevant code, callers, reachable execution paths, violated invariants, reproducible behavior, or a defensible complexity argument.
- Do not present a suspected defect as confirmed. State what has and has not been established, and identify the evidence needed to resolve the gap.
- Attempt to falsify each important finding. Look for counterexamples, unreachable paths, existing safeguards, relevant tests, and alternative explanations.
- Do not infer correctness from the absence of a reproduced failure or from tests that do not exercise the relevant behavior.
- Do not use uncertainty as a reason to stop investigating when source inspection, a targeted test, or a small experiment can resolve it.
- For performance claims, distinguish measured results from theoretical deductions and unmeasured hypotheses. Include workload, methodology, and limitations for measurements.
- Never claim to have run a test, benchmark, or analysis that was not actually performed.

Use **Verified**, **Derived**, **Suspected**, **Assumed**, and **Unknown** where useful, and explain the evidence or reasoning behind each consequential classification. A label alone is not evidence.

## 4. Required Finding Format

Every finding must contain:

### [Severity] Short, Specific Title

- **Location:** Exact file and relevant line range or symbol.
- **Category:** Correctness / Performance / Concurrency / Memory / Algorithm / Edge Case / Architecture.
- **Issue:** What is wrong or what opportunity exists.
- **Evidence:** Relevant execution path, invariant, complexity argument, or reproducible behavior.
- **Why it matters:** Concrete consequences.
- **Proposed change:** A technically specific improvement.
- **Expected impact:** Correctness improvement or anticipated performance benefit.
- **Complexity and trade-offs:** Implementation difficulty, additional memory, code size, maintenance cost, and other relevant costs.
- **Edge cases and risks:** Scenarios that must be considered.
- **Verification:** Tests, experiments, benchmarks, or reasoning needed to establish the result.

For optimization opportunities, explicitly distinguish measured improvements from theoretical or expected improvements.

## 5. Severity

Use severity consistently:

- **Critical:** Severe correctness or safety failure with substantial impact.
- **High:** Significant correctness, concurrency, or performance problem.
- **Medium:** Meaningful defect or optimization opportunity with a narrower impact.
- **Low:** Limited impact, minor defect, or relatively small optimization.

Do not assign severity solely based on implementation difficulty.

## 6. Review Output

Organize findings by severity, highest first. Separate:

1. Confirmed defects.
2. Potential defects requiring additional verification.
3. Algorithmic optimization opportunities.
4. Smaller performance improvements.
5. Missing tests and verification gaps.

Avoid padding the report with generic advice.

Conclude with:

- The most important findings.
- Areas examined without identified issues.
- Tests or experiments performed.
- Areas that could not be verified.
- Assumptions and environmental limitations.

## 7. Review Agent Prompt

Use this prompt when assigning a review task. Append the specific target, scope, and any relevant constraints.

> Review the assigned Ponent codebase or scope for correctness, performance, and edge cases. Prioritize concrete, specific issues and targeted improvement suggestions.
>
> In addition, actively identify algorithmic optimization opportunities—including high-effort or troublesome ones. Do not restrict the review to obvious defects or superficial micro-optimizations.
>
> Our codebase's discipline is that **the perfection of the project takes precedence over the comfort of its engineers**. Do not optimize for developer convenience at the expense of quality. Pursue aggressive, thorough optimization wherever technically justified, with the objective of maintaining our state-of-the-art capabilities.
>
> Investigate the relevant implementation, callers, callees, data structures, tests, and architectural invariants before reporting findings.
>
> For every finding, specify:
>
> - Location and affected code.
> - The precise issue or optimization opportunity.
> - Why it matters.
> - Evidence and reasoning.
> - The proposed change.
> - Expected impact.
> - Complexity and trade-offs.
> - Relevant edge cases and risks.
> - Required verification.
>
> Clearly distinguish confirmed defects from hypotheses, and measured improvements from unmeasured expectations.
>
> Do not modify source code. This is a review-only task. Deliver a technically rigorous report that can be handed directly to an implementation agent.
