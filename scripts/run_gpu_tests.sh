#!/usr/bin/bash
# Execute each line of gpu_test_files.txt as a shell command, sequentially.
# Designed for the Kaggle GPU box. NEVER runs tests in parallel.
# Continues on failure — every line is attempted; a report is printed at the end.
#
# gpu_test_files.txt holds one bare test file path per line, e.g.:
#     tests/test_abs.mojo
# Each line is run as: ./fire.sh <path>
#
# If gpu_test_files.txt does not exist it is auto-generated: every
# tests/test_*.mojo file (excluding chunk files, check_simd_width, and the
# names in SKIP_FILES below) that contains `has_accelerator()` is listed,
# matching run_gpu_test_files.sh.
#
# Usage:
#   ./run_gpu_tests.sh                        # run all lines
#   ./run_gpu_tests.sh --from test_sgd        # resume at test_sgd (after an interrupted run)
#   ./run_gpu_tests.sh --skip test_abs --skip test_sgd  # skip specific tests
#   ./run_gpu_tests.sh --sleep 0              # no pause between tests
#
# Env overrides:
#   LIST_FILE   list file to use (default: gpu_test_files.txt)
#
# Per-test logs go to logs/<test_name>.log; a summary is written to
# logs/kaggle_gpu_summary.txt and echoed to stdout at the end.
#
# NOTE: fire.sh invokes bare `mojo`, so run this from a `pixi shell`
# (or otherwise ensure `mojo` is on PATH).

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LIST_FILE="${LIST_FILE:-$SCRIPT_DIR/gpu_test_files.txt}"
LOG_DIR="$REPO_ROOT/logs"
SUMMARY_FILE="$LOG_DIR/kaggle_gpu_summary.txt"
mkdir -p "$LOG_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

# Test names (without .mojo) that are never run. Used both when generating
# the list and when executing lines from an existing list.
# test_attn_matmul is superseded by test_attn_matmul_cpu + test_attn_matmul_gpu.
declare -a SKIP_FILES=(
    test_attn_matmul
    test_gather
    test_exponential
    test_tanh
    test_logarithm
)

FROM_TEST=""
SLEEP_SECS=5
declare -a SKIP_NAMES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --from)
            FROM_TEST="$2"
            shift 2
            ;;
        --skip)
            SKIP_NAMES+=("$2")
            shift 2
            ;;
        --sleep)
            SLEEP_SECS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--from <test_name>] [--skip <test_name> ...] [--sleep N]"
            exit 1
            ;;
    esac
done

is_skipped() {
    local name="$1"
    for s in "${SKIP_FILES[@]}"; do
        [ "$name" = "$s" ] && return 0
    done
    for s in "${SKIP_NAMES[@]}"; do
        [ "$name" = "$s" ] && return 0
    done
    return 1
}

if [ ! -f "$LIST_FILE" ]; then
    echo -e "${YELLOW}$LIST_FILE not found — generating it from tests/test_*.mojo${NC}"
    generate_list() {
        : > "$LIST_FILE"
        for file in "$REPO_ROOT"/tests/test_*.mojo; do
            name=$(basename "$file" .mojo)
            case "$name" in
                test_cpu_all_*|test_gpu_all_*|check_simd_width)
                    continue
                    ;;
            esac
            if ! grep -q "has_accelerator()" "$file"; then
                continue
            fi
            if is_skipped "$name"; then
                continue
            fi
            echo "tests/$name.mojo" >> "$LIST_FILE"
        done
    }
    generate_list
    if [ ! -s "$LIST_FILE" ]; then
        echo -e "${RED}No GPU test files found — generated list is empty${NC}"
        exit 1
    fi
    echo -e "${GREEN}Generated $(wc -l < "$LIST_FILE") entries in $LIST_FILE${NC}"
fi

declare -a LINES=()
while IFS= read -r line; do
    [ -n "$line" ] && LINES+=("$line")
done < "$LIST_FILE"

if [ ${#LINES[@]} -eq 0 ]; then
    echo -e "${RED}$LIST_FILE is empty${NC}"
    exit 1
fi

# Derive a short name from a test file path (e.g. tests/test_abs.mojo -> test_abs).
line_name() {
    local line="$1"
    local m
    m=$(basename "$line" .mojo)
    if [ "$m" = "$line" ]; then
        # No .mojo suffix — fall back to the line as-is
        printf '%s\n' "$line"
    else
        printf '%s\n' "$m"
    fi
}

echo -e "${BOLD}Running ${#LINES[@]} GPU test commands sequentially${NC}"
echo -e "List file: $LIST_FILE"
echo -e "Sleep between tests: ${SLEEP_SECS}s (--sleep 0 disables)"
echo -e "Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

script_start=$(date +%s)
passed=0
failed=0
declare -a failed_names
started=false

idx=0
for line in "${LINES[@]}"; do
    idx=$((idx + 1))
    name=$(line_name "$line")

    if [ -n "$FROM_TEST" ]; then
        if [ "$started" = false ]; then
            if [ "$name" = "$FROM_TEST" ]; then
                started=true
            else
                continue
            fi
        fi
    fi

    if is_skipped "$name"; then
        echo -e "${YELLOW}SKIP${NC}   [$(date '+%H:%M:%S')] ($idx/${#LINES[@]}) $name"
        echo ""
        continue
    fi

    echo -e "${BOLD}[$(date '+%H:%M:%S')] ($idx/${#LINES[@]}) $name ...${NC}"

    logfile="$LOG_DIR/${name}.log"

    # ── Minute ticker (compiles take minutes) ────────────────────────────
    test_start=$(date +%s)
    (
        while true; do
            sleep 60
            now=$(date +%s)
            elapsed=$(( (now - test_start) / 60 ))
            echo -ne "\r  [${elapsed}m elapsed]"
        done
    ) &
    ticker_pid=$!

    if ( cd "$REPO_ROOT" && ./fire.sh "$line" ) > "$logfile" 2>&1; then
        kill "$ticker_pid" 2>/dev/null
        wait "$ticker_pid" 2>/dev/null
        elapsed=$(( ($(date +%s) - test_start) / 60 ))
        echo -e "\r  ${GREEN}PASS${NC}  (${elapsed}m)  "
        passed=$((passed + 1))
    else
        kill "$ticker_pid" 2>/dev/null
        wait "$ticker_pid" 2>/dev/null
        elapsed=$(( ($(date +%s) - test_start) / 60 ))
        echo -e "\r  ${RED}FAIL${NC}  (${elapsed}m)  — see $logfile"
        failed_names+=("$name")
        failed=$((failed + 1))
    fi
    echo ""

    if [ "$SLEEP_SECS" -gt 0 ]; then
        sleep "$SLEEP_SECS"
    fi
done

script_end=$(date +%s)
total_min=$(( (script_end - script_start) / 60 ))

echo "============================================"
echo -e "${BOLD}Results:${NC} ${GREEN}${passed} passed${NC}, ${RED}${failed} failed${NC} (${total_min}m total)"
if [ "$failed" -gt 0 ]; then
    echo -e "${RED}Failed:${NC}"
    for n in "${failed_names[@]}"; do
        echo "  - $n"
    done
fi
echo "============================================"
echo "End: $(date '+%Y-%m-%d %H:%M:%S')"

# Persist a copy of the summary for the Kaggle box
{
    echo "Kaggle GPU test run summary"
    echo "Run start : $(date -d @${script_start} '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date)"
    echo "Run end   : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Total     : ${passed} passed, ${failed} failed (${total_min}m)"
    if [ "$failed" -gt 0 ]; then
        echo "Failed    :"
        for n in "${failed_names[@]}"; do
            echo "  - $n"
        done
    fi
} > "$SUMMARY_FILE"
echo -e "${BOLD}Summary saved to $SUMMARY_FILE${NC}"

# Exit with failure count (0 if all passed)
exit "$failed"
