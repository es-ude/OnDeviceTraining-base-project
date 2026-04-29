#!/usr/bin/env bash
# Parity check: ODT mlp_mnist_float32_host (vendored ODT main) vs PyTorch.
# Builds with -O1 (matches the cloud-routine config), runs the N=10 harness
# sequentially, writes everything under runs/develop_check/.
#
# Usage: scripts/run-parity-check.sh [N]
#   N optional, overrides n_seeds for this run only (default uses whatever is
#   set in src/examples/reference/mlp_mnist_float32_host_ref.py — currently 10).
#
# Re-run safe: clears prior seed CSVs in runs/develop_check/ before starting
# so corrupted/partial files from earlier runs don't poison the comparison.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

OUT_DIR="runs/develop_check"
LOG="$OUT_DIR/_compare.log"

if [ ! -d "OnDeviceTraining/src/.git" ]; then
  echo "error: vendored ODT missing — run 'cmake --preset PREPARE' first." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

echo "==> clearing prior seed CSVs in $OUT_DIR"
find "$OUT_DIR" -maxdepth 1 -name 'mlp_mnist_float32_host_*' -delete

echo "==> configure + build with -O1"
cmake --preset HOST-Debug \
  -DODT_EXAMPLE=mlp_mnist_float32_host \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS_RELEASE="-O1 -DNDEBUG" \
  -DCMAKE_CXX_FLAGS_RELEASE="-O1 -DNDEBUG" >/dev/null
cmake --build --preset HOST-Debug -j

OPT=$(grep -m1 '"command"' build/HOST-Debug/compile_commands.json | tr ' ' '\n' | grep -m1 -E '^-O[0-9]')
echo "==> compile flag: $OPT (expected -O1)"
[ "$OPT" = "-O1" ] || { echo "error: -O1 not applied"; exit 1; }

echo "==> running harness (this is sequential — N seeds × 50 epochs × ODT + PyTorch)"
echo "==> log: $LOG"
echo "==> start: $(date)"

START=$SECONDS
RUNS_SUBDIR=develop_check PYTHONUNBUFFERED=1 \
  uv run src/examples/reference/mlp_mnist_float32_host_ref.py 2>&1 \
  | tee "$LOG"
HARNESS_EXIT=${PIPESTATUS[0]}
DUR=$((SECONDS - START))

echo "==> end: $(date)"
echo "==> wall-clock: ${DUR}s ($(printf '%dh%dm' $((DUR/3600)) $(( (DUR%3600)/60 ))))"
echo "==> harness exit: $HARNESS_EXIT (0=PASS, 1=FAIL)"
exit "$HARNESS_EXIT"
