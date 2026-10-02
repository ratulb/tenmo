#!/usr/bin/env bash
# scripts/run_python_tests.sh — Build _tenmo.so and run the Python test suite.
#
# Each test file runs in a separate subprocess to avoid JIT SIGILL corruption
# from accumulated backward passes in the same process.
#
# Usage:
#   ./scripts/run_python_tests.sh              # run all tests (isolated)
#   ./scripts/run_python_tests.sh --file tests/python/test_arithmetic.py  # single file
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SO="$REPO_ROOT/python-binding/_tenmo.so"
MOJO_SRC="$REPO_ROOT/python-binding/tenmo_bind.mojo"

cd "$REPO_ROOT"

# ── 1. Remove stale .so ──────────────────────────────────────────────
if [[ -f "$SO" ]]; then
    echo "── Removing stale $SO ..."
    rm -f "$SO"
fi

# ── 2. Build _tenmo.so ──────────────────────────────────────────────
# Memory cap prevents OOM on compile (peak ~15-17 GB with the f64 rich
# surface; was ~14 GB at two arith dtypes). -j2: default parallelism
# OOMs the binding build even under the cap. -O1: the full
# 50-method f64 surface OOMs at -O3 even under 17G; -O1 fits and the
# full suite is green on it.
echo "── Building $SO ..."
BUILD_CMD=(pixi run mojo build --optimization-level 1 -j2 "$MOJO_SRC" -I . --emit shared-lib -o "$SO")

if command -v systemd-run &>/dev/null && [[ "${EUID}" -ne 0 ]]; then
    systemd-run --scope --user \
        -p MemoryMax=17408M \
        "${BUILD_CMD[@]}"
else
    "${BUILD_CMD[@]}"
fi

echo "── Build succeeded ($(du -h "$SO" | cut -f1))"

# ── 3. Run tests ─────────────────────────────────────────────────────
# Parse --file flag for single-file mode
SINGLE_FILE=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --file)
            SINGLE_FILE="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -n "$SINGLE_FILE" ]]; then
    echo "── Running single file: $SINGLE_FILE ..."
    pixi run --environment dev python -m pytest "$SINGLE_FILE" -v --tb=short "${EXTRA_ARGS[@]}"
else
    echo "── Running each test file in isolated subprocess (avoids JIT SIGILL) ..."
    PASS=0
    FAIL=0
    SKIP=0
    TOTAL=0
    FAILED_FILES=()

    for f in tests/python/test_*.py; do
        name=$(basename "$f")
        TOTAL=$((TOTAL + 1))

        # Run each file in its own subprocess
        OUTPUT=$(pixi run --environment dev python -m pytest "$f" -v --tb=short "${EXTRA_ARGS[@]}" 2>&1) || true
        RC=$?

        # Extract summary line (guard: grep no-match + pipefail + set -e aborts)
        SUMMARY=$(echo "$OUTPUT" | grep -oE "[0-9]+ (passed|failed|errors?)" | tail -1 || true)

        if [[ $RC -eq 0 ]]; then
            echo "  PASS  $name  ($SUMMARY)"
            PASS=$((PASS + 1))
        elif [[ $RC -eq 5 ]]; then
            echo "  SKIP  $name  (no tests collected)"
            SKIP=$((SKIP + 1))
        else
            echo "  FAIL  $name  (rc=$RC)"
            echo "$OUTPUT" | grep -E "FAILED|ERROR" | head -5 | sed 's/^/        /' || true
            FAIL=$((FAIL + 1))
            FAILED_FILES+=("$name")
        fi
    done

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  Files: $TOTAL  Passed: $PASS  Failed: $FAIL  Skipped: $SKIP"
    if [[ $FAIL -gt 0 ]]; then
        echo "  Failed files: ${FAILED_FILES[*]}"
    fi
    echo "════════════════════════════════════════════════════════════"
    [[ $FAIL -eq 0 ]] && exit 0 || exit 1
fi
