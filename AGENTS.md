# Welcome, AI Agents

Welcome to Ponent.

You are encouraged to investigate difficult problems, challenge existing implementations, propose better algorithms, and undertake substantial engineering work.

Do not mistake the current implementation for the theoretical limit of what is possible.

At the same time, do not mistake an ambitious proposal for a correct solution.

**Understand deeply. Prove what matters. Measure what matters. Improve relentlessly.**

The goal is not to produce code quickly.

The goal is to make Ponent better.

## Engineering Philosophy

Our codebase operates under one fundamental principle:

**The perfection of the project takes precedence over the comfort of its engineers.**

Ponent aims to remain at the state-of-the-art frontier. Maintaining that position requires uncompromising correctness, rigorous engineering, and aggressive optimization wherever technically justified.

Do not favor an easier implementation merely because it is easier to write, understand, or maintain. When multiple approaches are available, prefer the one that provides stronger correctness guarantees, better performance, or a more capable architecture—provided its complexity and trade-offs are justified.

Do not introduce complexity for its own sake. Every optimization and architectural change must have a defensible technical rationale.

## 1. Engineering Standards

### Correctness

- Fix root causes rather than masking symptoms.
- Preserve established semantics and architectural invariants.
- Examine boundary conditions, failure paths, unusual inputs, and error handling.
- Pay particular attention to concurrency, synchronization, ownership, and object lifecycles.
- Never weaken assertions, tests, or type constraints merely to make an implementation pass.
- Distinguish confirmed defects from hypotheses and unverified assumptions.

### Epistemic Discipline

Confidence must be earned through evidence, not generated through language.

- Distinguish **observed facts**, **logical deductions**, **assumptions**, **hypotheses**, and **unknowns**. Do not collapse these into an undifferentiated claim.
- Support consequential claims with identifiable evidence: source locations, execution traces, test results, measurements, specifications, or explicit derivations.
- Keep conclusions proportional to the strength and scope of that evidence. A plausible explanation is not an established fact.
- When evidence is insufficient, state precisely what is missing and what would resolve the uncertainty. Do not manufacture confidence or hide uncertainty behind vague disclaimers.
- Do not use uncertainty as an excuse to avoid investigation. Resolve uncertainty whenever practical by inspecting code, tracing behavior, running experiments, or constructing counterexamples.
- Actively challenge your own conclusions. Search for counterexamples, test critical assumptions, and attempt to falsify proposed fixes and correctness arguments.
- Distinguish the absence of discovered problems from evidence that no problem exists.
- Never claim that code was tested, benchmarked, reviewed, or verified unless the corresponding operation was actually performed. Report the exact scope and outcome of verification.
- Keep unresolved risks, limitations, and unverified assumptions explicit, including their possible consequences.

Use these evidence labels in substantial investigations and reports where they improve precision:

| Label | Meaning |
|---|---|
| **Verified** | Directly checked through source evidence, execution, tests, or measurements. State which. |
| **Derived** | Follows logically from identified verified facts; show the reasoning. |
| **Suspected** | Supported by indications, but not yet established. |
| **Assumed** | Temporarily accepted to proceed; identify the assumption and its impact. |
| **Unknown** | The available evidence does not establish the answer. |

These labels are not a substitute for explanation or evidence. Do not label a claim **Verified** merely because it seems obvious or because an agent previously asserted it.

### Performance

Performance is an engineering requirement, not a cosmetic concern. Consider the complete cost model:

- Algorithmic complexity and asymptotic improvements.
- Allocation frequency, allocation size, and memory retention.
- Data layout, cache locality, and memory access patterns.
- Unnecessary cloning, copying, hashing, and synchronization.
- Lock contention, parallelism, and scheduling overhead.
- Incremental computation, caching, and reuse.
- Compile-time costs, monomorphization, and generated-code overhead.
- Worst-case behavior and adversarial inputs.

High-effort optimizations are welcome when their expected benefits justify their complexity. Do not claim performance improvements without measurements or a defensible theoretical argument.

### Design and Safety

- Prefer explicit ownership and lifetime designs over lifetime erasure.
- Prefer precise types and explicit invariants over loosely constrained abstractions.
- Avoid unnecessary allocations, indirection, dynamic dispatch, and synchronization.
- Consider compile-time costs and code size alongside runtime performance.
- Do not introduce `unsafe` without compelling technical necessity, a narrowly defined safety contract, and appropriate verification.
- Do not use `Box::leak`, `transmute`, raw-pointer lifetime reconstruction, or equivalent techniques as shortcuts around ownership and lifetime design problems.

## 2. Project Orientation

Ponent is a Rust compiler. The following map is an entry point, not a substitute for understanding the implementation.

| Path | Area |
|---|---|
| `src/ast/` | Abstract syntax tree |
| `src/lexer/` | Lexical analysis |
| `src/parser/` | Parsing |
| `src/hir/` | High-level intermediate representation |
| `src/hir/checker/` | Semantic checking |
| `src/hir/types/` | Type-system implementation |
| `src/hir/traits/` | Trait-related functionality |
| `src/hir/query/` | Query infrastructure |
| `src/mir/` | Mid-level intermediate representation |
| `src/sat/` | SAT-related functionality |
| `src/diagnostics/` | Diagnostics |
| `src/source_map/` | Source mapping |
| `src/arena.rs` | Arena infrastructure |
| `src/sync.rs` | Synchronization infrastructure |
| `src/loader.rs` | Loading infrastructure |
| `tools/codegen/` | Code-generation tools |
| `error_codes/` | Error-code definitions |
| `locales/` | Localization |
| `research/` | Research material |
| `benches/` | Benchmarks |

The compiler's main module graph is rooted through `src/main.rs`. Do not assume that `src/lib.rs` represents the complete compiler architecture.

Before changing a subsystem, inspect its actual implementation, callers, data ownership, tests, and relevant invariants.

### Particularly Sensitive Subsystems

**Query system:** The query infrastructure involves caching, dependency tracking, in-flight computation, synchronization, and lifecycle behavior. Analyze interactions among these mechanisms rather than treating queries as independent function calls.

**Type system and HIR:** Changes to type inference, bindings, type contexts, and HIR ownership can affect correctness across compilation units. Examine lifetime, unification, and sharing contracts.

**Synchronization and memory management:** Changes to shared state, allocation, reclamation, and concurrent execution must preserve safety and lifecycle guarantees.

**SAT and verification:** Changes to solver integration, constraints, and verification logic must preserve intended semantics and failure behavior.

## 3. Rust and Toolchain Requirements

- The repository MSRV is **Rust 1.92**.
- Do not introduce dependencies or language features that violate the declared MSRV.
- Prefer safe Rust and explicit ownership/lifetime designs.
- Preserve relevant `Send`, `Sync`, and other type-level contracts.
- Treat `unsafe` as a proof obligation, not a convenience.
- Consider monomorphization, code size, and compile-time overhead when designing abstractions.

## 4. Repository Workflow

### Understand Before Changing

Before modifying code:

1. Inspect the relevant implementation and its callers.
2. Identify existing invariants and architectural constraints.
3. Read related tests, benchmarks, and documentation.
4. Determine the scope and likely impact of the change.
5. Inspect the working tree and preserve pre-existing modifications.

Do not make architectural decisions based on isolated code fragments.

### Protect Existing Work

- Never discard, overwrite, or revert unrelated user changes.
- Never reset the working tree or perform destructive Git operations without explicit authorization.
- Do not overwrite another agent's in-progress work.
- Keep changes scoped and reviewable.
- Do not commit, push, create pull requests, or merge unless explicitly authorized.

### Verification

After modifying code:

- Run relevant tests.
- Run formatting and lint checks where appropriate.
- Check affected feature configurations.
- Verify boundary conditions and regression scenarios.
- Run relevant benchmarks when performance is involved.
- Inspect the final diff for unintended changes.

Ponent has distinct testing configurations. Ordinary tests and Shuttle-based concurrency tests are not interchangeable. Useful test entry points include:

```bash
cargo test --workspace --features omniml_pool,slow-tests
cargo test --features shuttle
```

Consult the current CI configuration and relevant documentation before choosing additional commands. Do not assume that `--all-features` is an adequate substitute for dedicated Shuttle testing.

Report exactly what was executed and what passed. Never claim that code was tested, benchmarked, or verified when it was not.

## 5. Working with Other Agents

- Inspect the current working tree before making changes.
- Respect changes made by other agents.
- Avoid unnecessary overlap in files and responsibilities.
- Communicate assumptions, dependencies, and architectural constraints.
- Do not silently replace another agent's work.
- Keep implementation and verification results traceable.

A review task and an implementation task are different responsibilities. Do not silently turn a review-only task into an implementation task.

## 6. Specialized Workflows

The detailed procedures and reusable task prompts are maintained separately:

- **Code review:** [`docs/agents/review.md`](docs/agents/review.md)
- **Implementation:** [`docs/agents/implementation.md`](docs/agents/implementation.md)

These protocols supplement this document. They do not override its engineering principles or repository protections.

## 7. Reporting and Technical Honesty

Distinguish clearly between:

- Confirmed defects and potential defects.
- Proven algorithmic improvements and optimization hypotheses.
- Measured performance improvements and theoretical expectations.
- Completed work and partially implemented work.
- Verified behavior and behavior that could not be tested.

For every substantial change, explain its rationale, impact, trade-offs, and verification. Do not inflate findings, invent benchmark results, or hide unresolved problems.

**The objective is not merely to make Ponent work. The objective is to make Ponent exceptionally good.**
