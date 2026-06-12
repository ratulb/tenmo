#!/usr/bin/bash
# Run all GPU-stripped test files from tests/gpu/
# Usage: ./run_gpu_stripped.sh
# Run from repo root: `mojo -I . tests/gpu/<file>.mojo`

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
BOLD='\033[1m'
NC='\033[0m'

passed=0
failed=0
declare -a failed_files

for file in "$SCRIPT_DIR"/tests/gpu/test_*.mojo; do
    name=$(basename "$file" .mojo)

    echo -e "${BOLD}[$(date '+%H:%M:%S')] Running ${name} ...${NC}"

    logfile="$LOG_DIR/${name}.log"

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

    if time mojo -I . "$file" > "$logfile" 2>&1; then
        kill "$ticker_pid" 2>/dev/null
        wait "$ticker_pid" 2>/dev/null
        elapsed=$(( ($(date +%s) - test_start) / 60 ))
        echo -e "\r  ${GREEN}PASS${NC}  (${elapsed}m)"
        passed=$((passed + 1))
    else
        kill "$ticker_pid" 2>/dev/null
        wait "$ticker_pid" 2>/dev/null
        elapsed=$(( ($(date +%s) - test_start) / 60 ))
        echo -e "\r  ${RED}FAIL${NC} (${elapsed}m) — see $logfile"
        failed_files+=("$name")
        failed=$((failed + 1))
    fi

    # ── Estimate remaining time ──────────────────────────
    if [ "$passed" -gt 0 ] || [ "$failed" -gt 0 ]; then
        total_done=$((passed + failed))
        avg_min=$(( ($(date +%s) - test_start) / total_done / 60 ))
        remaining=$(( (55 - total_done) * avg_min ))
        echo "  ${total_done}/55 done, ~${remaining}m remaining"
    fi
done

echo "============================================"
echo -e "${BOLD}Results:${NC} ${GREEN}$passed passed${NC}, ${RED}$failed failed${NC}"
if [ "$failed" -gt 0 ]; then
    echo -e "${RED}Failed:${NC} ${failed_files[*]}"
fi
echo "============================================"

exit "$failed"
