#!/usr/bin/bash
# Phase-2 playbook: warm-start a fresh Colab/Kaggle session.
#
#   - installs pixi deps
#   - (optional) persists the Mojo compile cache on Drive
#   - precompiles the tenmo package once  (tests/tenmo.mojoc, gitignored)
#       ^ packaging, NOT compilation: a .mojoc stores the library FRONTEND
#         (parse/sema). Measured NEUTRAL on a GPU box (~19.7s both ways on a
#         cold small program) — the per-program comptime/codegen cost is not
#         reduced by it. Cheap insurance, NOT a solution to cold GPU compile
#         times.
#   - generates the chunked GPU/CPU suites
#   - prints the SEQUENTIAL run commands
#
# NEVER run the chunk files in parallel — each chunk is a fresh `mojo`
# process; concurrent compiles OOM the machine. `./execute.sh
# gpu_all ...` is sequential by default; never pass `-p`.
#
# The real lever is reducing per-program comptime (Phase 3) and fewer programs
# (aggregation): keep CHUNKS small (default 4, max 8). More chunks does NOT
# speed up a cold box — each chunk pays per-program frontend + comptime.
#
# Usage:
#   ./scripts/colab_setup.sh
#   COLAB_CACHE_DIR=/content/drive/MyDrive/mojo_cache ./scripts/colab_setup.sh
#
# Env overrides:
#   COLAB_CACHE_DIR   Drive dir to persist the compile cache into (Colab)
#   CHUNKS            chunk count (default 4; keep 4-8)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_DIR="$REPO/.pixi/envs/default"
CACHE_DIR="$ENV_DIR/share/max/cache"
CHUNKS="${CHUNKS:-4}"

cd "$REPO"

echo "==> pixi install"
pixi install

# --- Optional: persist compile cache on Drive (Colab) ---
if [ -n "${COLAB_CACHE_DIR:-}" ]; then
    mkdir -p "$COLAB_CACHE_DIR"
    if [ -L "$CACHE_DIR" ]; then
        echo "==> cache already symlinked: $CACHE_DIR -> $COLAB_CACHE_DIR"
    else
        echo "==> symlinking $CACHE_DIR -> $COLAB_CACHE_DIR"
        rm -rf "$CACHE_DIR"
        ln -s "$COLAB_CACHE_DIR" "$CACHE_DIR"
    fi
else
    echo "==> compile cache left in place ($CACHE_DIR); set COLAB_CACHE_DIR to persist it"
fi

# --- One library build ---
echo "==> precompiling package -> tests/tenmo.mojoc (gitignored)"
pixi run mojo precompile -o tests/tenmo.mojoc tenmo/

# --- Generate chunked suites ---
echo "==> generating $CHUNKS GPU chunks"
pixi run python3 scripts/generate_gpu_test_suite.py --chunks "$CHUNKS"
echo "==> generating $CHUNKS CPU chunks"
pixi run python3 scripts/generate_cpu_test_suite.py --chunks "$CHUNKS"

run_list=$(seq -s ' ' 1 "$CHUNKS")

cat <<EOF

==> Run (SEQUENTIAL — never use -p):

    ./execute.sh gpu_all $run_list
    ./execute.sh cpu_all $run_list

Each chunk is one sequential mojo process. Do NOT run them in parallel;
concurrent compiles OOM the machine.
EOF
