#!/usr/bin/env bash
# Deterministic Ir/get wrapper for the query perf probes.
#
# The repo's wall-clock probes say this box drifts >2x between
# invocations; sub-10% deltas are gated on callgrind Ir.  This script
# runs the same ignored test twice under callgrind:
#
#   1. PONENT_BENCH_IR=warmup  — build + warm the graph, no measured pass
#   2. PONENT_BENCH_IR=<mode>  — one measured pass
#
# and reports (Ir(measured) - Ir(warmup)) / gets.  The subtraction
# removes the test harness, graph construction, and warmup, so the
# result is the measured pass's deterministic instruction count per
# operation.
#
# Usage:
#   tools/bench-query-ir.sh [test_name] [mode]
#
#   test_name  default: bench_invalidation_storm
#   mode       default: storm
#              bench_invalidation_storm: hit | storm
#              bench_fanout_publish:     fanout_control | fanout_change |
#                                        fanout_readers | fanout
#
# For bench_fanout_publish, subtract fanout_control from fanout_change to
# isolate the fan-out loop, and subtract fanout_change from fanout_readers
# to isolate the downstream republish pass.
#
# Environment:
#   PROFILE=release|dev   default: release
#   OUT_DIR=...           default: target/callgrind
set -euo pipefail

TEST="${1:-bench_invalidation_storm}"
MODE="${2:-storm}"
PROFILE="${PROFILE:-release}"
OUT_DIR="${OUT_DIR:-target/callgrind}"

if ! command -v valgrind >/dev/null 2>&1; then
    echo "error: valgrind not found (install valgrind / callgrind)" >&2
    exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "error: python3 not found (needed to parse cargo's JSON artifacts)" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

CARGO_PROFILE_FLAG=()
if [[ "$PROFILE" == "release" ]]; then
    CARGO_PROFILE_FLAG=(--release)
else
    CARGO_PROFILE_FLAG=(--profile "$PROFILE")
fi

# Build the unit-test binary once and pull its path out of cargo's JSON
# stream.  The root package is a bin target, so its test harness is the
# artifact with profile.test == true and target.name == "ponent".
BIN="$(
    cargo test "${CARGO_PROFILE_FLAG[@]}" --bins --no-run --message-format=json 2>/dev/null |
        python3 -c '
import json, sys
for line in sys.stdin:
    try:
        m = json.loads(line)
    except json.JSONDecodeError:
        continue
    if m.get("reason") != "compiler-artifact":
        continue
    target = m.get("target", {})
    profile = m.get("profile", {})
    if target.get("name") == "ponent" and profile.get("test") and m.get("executable"):
        print(m["executable"])
' | tail -1
)"
if [[ -z "$BIN" || ! -x "$BIN" ]]; then
    echo "error: could not locate the ponent test binary" >&2
    exit 1
fi

run_one() {
    local mode="$1" out="$2"
    PONENT_BENCH_IR="$mode" valgrind \
        --tool=callgrind \
        --callgrind-out-file="$out" \
        --quiet \
        "$BIN" --ignored --nocapture --test-threads=1 "$TEST" \
        2>&1 | tee "$out.log"
}

WARM_OUT="$OUT_DIR/$TEST.warmup"
MEASURED_OUT="$OUT_DIR/$TEST.$MODE"
run_one warmup "$WARM_OUT"
run_one "$MODE" "$MEASURED_OUT"

ir_of() {
    # callgrind writes a final "summary: <Ir>" line to the output file.
    awk '/^summary:/ { print $2; exit }' "$1"
}

IR_WARM="$(ir_of "$WARM_OUT")"
IR_MEASURED="$(ir_of "$MEASURED_OUT")"
GETS="$(
    grep -oE 'gets=[0-9]+' "$MEASURED_OUT.log" |
        tail -1 |
        cut -d= -f2
)"

if [[ -z "$IR_WARM" || -z "$IR_MEASURED" || -z "$GETS" || "$GETS" == "0" ]]; then
    echo "error: missing Ir/gets (warm=$IR_WARM measured=$IR_MEASURED gets=$GETS)" >&2
    exit 1
fi

DELTA=$((IR_MEASURED - IR_WARM))
awk -v w="$IR_WARM" -v m="$IR_MEASURED" -v d="$DELTA" -v g="$GETS" -v t="$TEST" -v mode="$MODE" '
BEGIN {
    printf "%s (%s): Ir(warmup)=%s Ir(measured)=%s delta=%s gets=%s Ir/get=%.2f\n",
        t, mode, w, m, d, g, d / g
}
'
