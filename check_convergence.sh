#!/bin/bash
# Run examples/reverse_sequence.mojo N times and verify convergence.
# Usage: ./check_convergence.sh [count=100]

set -euo pipefail

COUNT="${1:-100}"
PASS=0
FAIL=0
FAILED_RUNS=""

for i in $(seq 1 "$COUNT"); do
    echo -n "Run $i/$COUNT ... "
    output=$(mojo -I . examples/reverse_sequence.mojo 2>&1)

    acc_line=$(echo "$output" | grep "Best sequence accuracy")
    acc=$(echo "$acc_line" | sed 's/.*: \([0-9.]*\) .*/\1/')
    if [ -z "$acc" ]; then
        echo "FAIL (could not parse accuracy)"
        FAIL=$((FAIL + 1))
        FAILED_RUNS="$FAILED_RUNS $i"
    elif python3 -c "exit(0 if float('$acc') >= 99.0 else 1)" 2>/dev/null; then
        echo "OK (${acc}%)"
        PASS=$((PASS + 1))
    else
        echo "FAIL (${acc}%)"
        FAIL=$((FAIL + 1))
        FAILED_RUNS="$FAILED_RUNS $i"
    fi
done

echo ""
echo "=== Results ==="
echo "Passed: $PASS / $COUNT"
echo "Failed: $FAIL / $COUNT"
if [ "$FAIL" -gt 0 ]; then
    echo "Failed runs:$FAILED_RUNS"
    exit 1
fi
