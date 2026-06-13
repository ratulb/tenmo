#!/usr/bin/bash
# Run GPU-stripped test files from tests/gpu/
# Usage:
#   ./run_gpu_stripped.sh                          # run all
#   ./run_gpu_stripped.sh --ignore f1.mojo f2.mojo # skip specific files
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

# ── Parse --ignore flag ────────────────────────────────────────────
ignore_stems=()
if [[ "$1" == "--ignore" ]]; then
    shift
    while [[ "$#" -gt 0 ]]; do
        raw="$(basename "$1")"
        raw="${raw%.mojo}"
        ignore_stems+=("$raw")
        shift
    done
fi

ticker_pid=""
overall_start=$(date +%s)
file_count=0
declare -a file_list
for f in "$SCRIPT_DIR"/tests/gpu/test_*.mojo; do
    [ -f "$f" ] || continue
    stem="$(basename "$f" .mojo)"
    skip=false
    for ign in "${ignore_stems[@]}"; do
        if [[ "$stem" == "$ign" ]]; then
            skip=true
            break
        fi
    done
    $skip && continue
    file_list+=("$f")
done
file_count=${#file_list[@]}

if [ "${#ignore_stems[@]}" -gt 0 ]; then
    echo -e "${BOLD}Ignoring ${#ignore_stems[@]} file(s):${NC} ${ignore_stems[*]}"
fi
echo -e "${BOLD}Queued ${file_count} file(s)${NC}"

cleanup() {
    echo -e "\n${BOLD}Interrupted — cleaning up...${NC}"
    [ -n "$ticker_pid" ] && kill "$ticker_pid" 2>/dev/null
    exit 1
}
trap cleanup SIGINT SIGTERM

for file in "${file_list[@]}"; do
    name=$(basename "$file" .mojo)

    echo -e "${BOLD}[$(date '+%H:%M:%S')] Running ${name} ...${NC}"

    logfile="$LOG_DIR/${name}.log"
    test_start=$(date +%s)

    # Background ticker — shows elapsed time every 60s
    (
        while true; do
            sleep 60
            now=$(date +%s)
            elapsed=$(( (now - test_start) / 60 ))
            echo -ne "\r  [${elapsed}m elapsed]"
        done
    ) &
    ticker_pid=$!

    # Run mojo in foreground — Ctrl+C kills it directly
    mojo -I . "$file" > "$logfile" 2>&1
    mojo_exit=$?

    kill "$ticker_pid" 2>/dev/null
    wait "$ticker_pid" 2>/dev/null
    ticker_pid=""

    elapsed=$(( ($(date +%s) - test_start) / 60 ))

    if [ "$mojo_exit" -eq 0 ]; then
        echo -e "\r  ${GREEN}PASS${NC}  (${elapsed}m)"
        passed=$((passed + 1))
    else
        echo -e "\r  ${RED}FAIL${NC} (${elapsed}m) — see $logfile"
        failed_files+=("$name")
        failed=$((failed + 1))
    fi

    total_done=$((passed + failed))
    if [ "$total_done" -gt 0 ] && [ "$overall_start" -ne 0 ]; then
        elapsed_total=$(( ($(date +%s) - overall_start) / 60 ))
        [ "$elapsed_total" -lt 1 ] && elapsed_total=1
        avg_min=$(( elapsed_total / total_done ))
        remaining=$(( (file_count - total_done) * avg_min ))
        echo "  ${total_done}/${file_count} done, ~${remaining}m remaining"
    fi
done

echo "============================================"
echo -e "${BOLD}Results:${NC} ${GREEN}$passed passed${NC}, ${RED}$failed failed${NC}"
if [ "$failed" -gt 0 ]; then
    echo -e "${RED}Failed:${NC} ${failed_files[*]}"
fi
echo "============================================"

exit "$failed"
