# Shuttle migration — Phase 1/2 record

Architecture: **dual-track (A)**. Two test lanes, one crate:

| Lane | Command | Contents |
|---|---|---|
| default | `cargo test` | full plain suite (1585 tests) unchanged; shuttle tests compiled out |
| feature | `cargo test --features shuttle` | ONLY `shuttle_tests` (12 `#[test]`s, all `run_check`-wrapped); plain test mods gated `#[cfg(all(test, not(feature = "shuttle")))]` (57 mod decls) |

CI runs both: `.github/workflows/ci.yml` must NOT use `--all-features` for `cargo test`
(enabling `shuttle` would drop the plain suite to 0 tests) — it runs
`cargo test --workspace --features omniml_pool,slow-tests` plus a separate
`cargo test --features shuttle`.

## Feature-lane suite (12 tests)

- `shuttle_tests/b1.rs` — d10 (side-door invalidate vs publish), d17 (reentrant invalidate), d11 (reentrant invalidate inside compute), d15 (clear_all vs quiesced reader)
- `shuttle_tests/b2.rs` — d19 (clear must quiesce in-flight publish)
- `shuttle_tests/b3.rs` — d21 (staged value vs concurrent probe), d23 (per-thread stat merge)
- `shuttle_tests/depgraph.rs` — d26 (late durability recheck), d27 (seqlock vs inflight lowering)
- `shuttle_tests/aclass.rs` — d5 (eval_always never dedups; sequential form), d9 (clear_all hygiene; windows 2/3)
- `shuttle_tests/tstress.rs` — stress oracle (sequential form, 200 ops)

Scaffolding: `shuttle_tests/mod.rs` `run_check` = custom `Runner` with
`PctScheduler::new(8, 50)` and **2 MiB stack** (shuttle's default coroutine stack is
60 KiB — `DepGraph::new`'s 32 KiB frame SIGSEGVs without the bump; never use
`shuttle::check_pct` directly here). Single-threaded tests spawn a STOP-flag churn
thread so PCT sees `runnable.len() > 1`. Failure persistence:
`config.failure_persistence = FailurePersistence::File(None)` → failing runs write
`scheduleNNN.txt` at the crate root (gitignored).

## Unportable tests (skipped in the feature lane; plain lane keeps coverage)

Shuttle runs everything on **one carrier thread**. `std::thread::current().id()` is the
same `ThreadId` for every task, and `thread_local!` state is shared across tasks.

1. **Cross-task latch parking → false cycle.** `QuerySystem::register_wait` walks the
   wait graph with `me = current().id()` and errors on `cur == me` on the first
   iteration because owner == waiter == carrier. Any test whose core asserts that a
   second task *parks* on a first task's latch gets an immediate `QueryCycleError`.
   Skipped: **d18** (waiter re-resolves after clear_all), **d24** (green bit vs
   in-flight lowering), **d7** (parallel same-key dedup), **d5 overlap form** (kept as
   sequential variant), **t2** (staging panic releases waiters), **t1** (mutual
   request storm must never cycle — false cycles make every get err).
2. **Concurrent compute frames → `TASK_STACK` corruption.** `DepGraph::start_task`
   pushes onto a `thread_local!` frame stack and `finish_task_and_publish` pops it;
   two interleaved computes on one carrier pop each other's frames ("popped task
   frame does not belong to node") and `read()` misattributes dep edges. Skipped:
   **d3** (two-thread mutual dependency); stress runs with one inline worker instead
   of four.
3. **Wall-clock load-bearing.** **d9 window 1** depends on a 2s `Instant`
   self-timeout inside a parked waiter — removed with the parking; windows 2/3 ported.
4. **d19 weakened-assert note:** the plain test's post-clear `K_RUNS == 2` relied on
   parking_lot writer preference to order clear between owner-guard-drop and waiter
   re-take. The port instead relies on the product fix below; the wait-graph parking
   half of the original choreography is not reproduced (that aspect is d18-class).

## Product fixes made during migration

- `mutate_world` defer path (query/mod.rs): after pushing a deferred mutation, if
  `IN_FLIGHT_TOTAL` (per-thread) is 0 the mutation is drained immediately. Under
  shuttle's shared TLS a clearer can yield inside `deferred.lock()` after the last
  frame drained, orphaning the closure (observed: d19 post-clear serve of stale
  entry). Condition must be thread-TLS, **not** the per-system `frames` counter:
  an `frames == 0` variant prematurely drained mutations parked against a *different*
  live system and broke d22 in the plain lane.
- `stat_bump_cold` (query/mod.rs): no longer holds a `STAT_SLOTS.borrow_mut()` across
  the `stat_shards.lock()` call — under shuttle the lock is a yield point and a
  concurrent fast-path `borrow()` panicked (`RefCell already mutably borrowed`).
  Duplicate slot registration on a race is benign: `stats()` sums all slots.
- `DepGraph::lower_and_propagate` widened to `pub(crate)` (needed by d26/d27 ports).
- `src/sync.rs` dual lane: parking_lot re-export by default, newtype wrappers over
  `shuttle::sync` under the feature. `scheduler.rs` `PARK_TIMEOUT`/`Instant` deadlines
  cfg'd out under the feature — parking blocks forever and shuttle's deadlock
  detection (max-steps) is the liveness backstop.

## C-class / loom division of labor

- **C-class (rayon / `par_get` heavy tests): plain lane only.** The feature lane never
  builds those test mods; rayon worker pools are real OS threads and cannot run inside
  a shuttle check. No porting planned — contention behaviour is covered by shuttle's
  own thread interleavings in the ported tests, and parallel-throughput regressions are
  a plain-lane concern.
- **loom: not adopted.** Shuttle subsumes the synchronization coverage we would write
  loom tests for (locks, atomics, parking), with the ability to run the *real*
  product code rather than loom-modelled abstractions. loom would additionally require
  its own cfg'd type universe. Revisit only if a specific atomic protocol needs
  exhaustive small-state exploration.
- **Wall-clock / timeout logic: plain lane.** Deadlock-timeout paths
  (`PARK_TIMEOUT`, guard deadlines) are compiled out under shuttle by design
  (spec §5); their behaviour is asserted by the plain suite.

## Replay flow

1. Failing feature-lane run persists `scheduleNNN.txt` (cwd) and prints the path.
2. Reproduce deterministically with `shuttle::replay_from_file("scheduleNNN.txt")`,
   or paste the hex printed by `FailurePersistence::Print` into
   `shuttle::replay("...")`. For local debugging, swap `run_check` to
   `ReplayScheduler::new_from_file(...)` (shuttle-schedulers).

## Warning budget

- default `cargo check --tests`: 552 warnings (pre-shuttle baseline, must not grow)
- feature `cargo check --tests --features shuttle`: 559 (+7 — plain-test-only helpers
  orphaned by gating the 5195-line defense mod; only `sync.rs::try_read` was actually
  dead from our own changes and was removed. No `#[allow]`s.)
