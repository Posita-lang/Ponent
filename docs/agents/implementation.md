# Implementation Protocol

This document defines the implementation standards and workflow for AI agents resolving review findings and improving Ponent.

The objective is not merely to make reported findings disappear. It is to deliver the strongest technically justified implementation, preserve architectural invariants, and verify the resulting changes.

## 1. Implementation Mission

Given a set of review findings, implement the necessary correctness fixes and justified optimizations.

The scope includes:

- Resolving confirmed defects.
- Investigating and resolving potential defects where evidence supports a fix.
- Implementing algorithmic improvements.
- Improving memory efficiency and allocation behavior.
- Improving concurrency and synchronization behavior.
- Strengthening invariants and regression coverage.
- Addressing relevant edge cases.
- Verifying that changes do not introduce regressions.

Do not interpret the findings as an exhaustive list of all possible improvements. If implementation reveals additional important defects or optimization opportunities within the assigned scope, investigate and address them where justified.

## 2. Implementation Principle

Our codebase operates under one fundamental principle:

**The perfection of the project takes precedence over the comfort of its engineers.**

Achieving and maintaining this standard demands relentless optimization and uncompromising correctness fixes to ensure that Ponent remains at the state-of-the-art frontier.

**Whenever there is a choice, choose the more thorough approach—the one that pushes us closer to SOTA, no matter how much effort it requires.**

This requires technical judgment:

- Fix root causes, not symptoms.
- Prefer stronger correctness guarantees.
- Pursue meaningful algorithmic improvements.
- Do not settle for superficial workarounds.
- Do not avoid difficult work when it is justified.
- Do not introduce complexity without a defensible technical benefit.

## 3. Validate Before Implementing

For every finding:

1. Locate the affected implementation.
2. Reconstruct the relevant execution path.
3. Verify the finding against the actual code.
4. Check whether the issue has already been resolved.
5. Examine relevant invariants and callers.
6. Evaluate the proposed fix and its trade-offs.
7. Determine whether a more complete solution exists.

Do not blindly implement an incorrect or incomplete review recommendation.

If a finding is invalid, already fixed, or based on an incorrect assumption, document the reason rather than introducing an unnecessary change.

### Epistemic Discipline

Implementation decisions must be grounded in evidence, not confidence expressed through language.

- Distinguish **observed facts**, **logical deductions**, **assumptions**, **hypotheses**, and **unknowns** when evaluating findings and choosing a solution.
- Independently verify each consequential review claim against the current code. A review finding is an input to investigate, not an instruction to trust.
- Identify assumptions on which the implementation depends. Resolve them where practical; otherwise document their scope and consequences.
- Attempt to falsify the proposed solution before accepting it: examine counterexamples, boundary conditions, failure paths, callers, and interactions with architectural invariants.
- Do not mistake a passing test for proof beyond the behavior that test actually covers. Add targeted regression tests or other verification for the relevant claim.
- Do not use uncertainty as an excuse to defer investigation that can reasonably be performed.
- Report exactly what was changed and what was verified. Separate verified outcomes from deductions, expectations, and unresolved risks.
- Never claim a test, benchmark, or verification step succeeded unless it was actually executed and its result observed.

Use **Verified**, **Derived**, **Suspected**, **Assumed**, and **Unknown** where useful in implementation reports. Explain the supporting evidence; do not use labels as decoration.

## 4. Implementation Requirements

### Correctness

- Address the underlying cause.
- Preserve existing semantics unless a deliberate change is required.
- Handle boundary conditions and failure paths.
- Maintain concurrency and synchronization invariants.
- Add regression tests for fixed defects.
- Do not weaken tests, assertions, or type constraints to accommodate an incorrect implementation.

### Optimization

Consider the strongest viable solution, including algorithmic redesign where appropriate. Evaluate:

- Time and space complexity.
- Allocation and deallocation behavior.
- Cache locality and data layout.
- Unnecessary cloning, copying, and hashing.
- Lock contention and synchronization overhead.
- Parallel execution and scheduling.
- Cache reuse and incremental computation.
- Compile-time overhead and code size.

Do not stop at a local optimization if a more substantial algorithmic improvement is justified.

### Rust-Specific Constraints

- Respect the repository's Rust 1.92 MSRV.
- Prefer safe Rust and explicit ownership/lifetime designs.
- Do not use `Box::leak`, `transmute`, raw-pointer lifetime reconstruction, or equivalent lifetime-erasure shortcuts to bypass design problems.
- Introduce `unsafe` only when necessary and supported by a precise safety argument.
- Preserve `Send`, `Sync`, and other relevant type-level contracts.
- Consider monomorphization and compile-time costs alongside runtime performance.

## 5. Testing and Verification

Every implementation must be accompanied by appropriate verification. Depending on the affected subsystem, consider:

- Unit tests.
- Integration tests.
- Regression tests.
- Property-based tests.
- Concurrency tests.
- Shuttle tests.
- Feature-specific test configurations.
- Benchmarks.
- Formatting and lint checks.

For query-system or synchronization changes, do not assume ordinary unit tests are sufficient. Examine the relevant concurrency scenarios and dedicated test configurations.

For performance changes:

- Establish a meaningful baseline where possible.
- Use representative workloads.
- Measure the relevant operation.
- Compare equivalent configurations.
- Report measurement methodology and limitations.
- Avoid claiming improvements based on unverified intuition.

Never weaken the verification standard to make an implementation appear complete.

## 6. Protect the Repository

- Preserve unrelated working-tree modifications.
- Do not overwrite other agents' changes.
- Avoid broad or unrelated refactoring.
- Do not discard work through destructive Git operations.
- Keep the diff focused and reviewable.
- Do not commit, push, create pull requests, or merge without explicit authorization.

## 7. Completion Report

For every finding, report one of:

- **Resolved:** Implemented and verified.
- **Partially resolved:** Explain what remains.
- **Rejected:** Explain why the finding was invalid or the proposed change was unjustified.
- **Deferred:** Explain the blocker and the remaining work.

For every implementation, report:

- Files and symbols changed.
- Root cause and implemented solution.
- Tests added or modified.
- Commands executed and their results.
- Benchmarks and measurements, if applicable.
- Remaining risks and limitations.
- Additional issues discovered.

Be explicit about what was not verified.

## 8. Implementation Agent Prompt

Append the relevant review findings after this prompt when assigning an implementation task:

> Resolve every valid finding and implement every justified optimization in the supplied review report.
>
> Our codebase operates under one fundamental principle: **the perfection of the project takes precedence over the comfort of its engineers.** Achieving and maintaining perfection demands relentless, almost obsessive optimization and uncompromising correctness fixes to ensure that we remain at the SOTA frontier.
>
> **Whenever there is a choice, choose the more thorough approach—the one that pushes us closer to SOTA, no matter how much effort it requires.**
>
> Do not blindly follow the proposed fixes. Independently validate every finding, investigate its root cause, and choose the strongest technically justified solution. A superficial workaround is not an acceptable substitute for a complete fix.
>
> Actively investigate additional correctness issues and algorithmic optimization opportunities exposed by the implementation work. Do not limit yourself to the exact lines mentioned in the findings.
>
> Preserve all relevant architectural, ownership, lifetime, concurrency, and synchronization invariants. Add appropriate regression tests and verify the affected behavior.
>
> Do not weaken tests, assertions, type constraints, or safety guarantees to make the implementation pass.
>
> Measure performance improvements where practical. Clearly distinguish measured results from theoretical expectations.
>
> Preserve unrelated working-tree changes and do not perform destructive Git operations.
>
> At completion, provide a finding-by-finding report, including implementation details, verification results, benchmarks where applicable, and all remaining limitations.
