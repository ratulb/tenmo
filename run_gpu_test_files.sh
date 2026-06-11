#!/usr/bin/bash
# Run every GPU-containing test file individually via `mojo`.
# Uses precompiled package: tests/tenmo.mojopkg
#   (run `mojo package -o tests/tenmo.mojopkg tenmo/` first)
# Does NOT stop on failure — logs results for all files.
#
# Usage: ./run_gpu_test_files.sh

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

for file in "$SCRIPT_DIR"/tests/test_*.mojo; do
    name=$(basename "$file" .mojo)

    # Skip generated chunk files
    case "$name" in
        test_cpu_all_*|test_gpu_all_*|check_simd_width)
            continue
            ;;
    esac

    # Skip files with no GPU tests
    if ! grep -q "has_accelerator()" "$file"; then
        continue
    fi

    echo -e "${BOLD}[$(date '+%H:%M:%S')] Running ${name} ...${NC}"

    logfile="$LOG_DIR/${name}.log"
    if time mojo "$file" > "$logfile" 2>&1; then
        echo -e "  ${GREEN}PASS${NC}"
        passed=$((passed + 1))
    else
        echo -e "  ${RED}FAIL${NC} — see $logfile"
        failed_files+=("$name")
        failed=$((failed + 1))
    fi
    echo ""
done

echo "============================================"
echo -e "${BOLD}Results:${NC} ${GREEN}$passed passed${NC}, ${RED}$failed failed${NC}"
if [ "$failed" -gt 0 ]; then
    echo -e "${RED}Failed:${NC} ${failed_files[*]}"
fi
echo "============================================"

# Exit with failure count (0 if all passed)
exit "$failed"
