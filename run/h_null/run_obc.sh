#!/usr/bin/env bash
# run_obc.sh — OBC sweep, h=0
# Usage: bash run/h_null/run_obc.sh
# Can be run from any directory.

set -euo pipefail
cd "$(dirname "$0")/../.."

BINARY="./ising_static"
H="0.0"
PBC="0"

echo "=============================================="
echo "  OBC production run"
echo "  L: 4 6 8 10 12  (src/h_null/main_static.c)"
echo "=============================================="

[[ -x "${BINARY}" ]] || { echo "[ERROR] '${BINARY}' not found. Run 'make' first."; exit 1; }

mkdir -p data/h_null/observables/OBC

echo "[RUN] ${BINARY} ${H} ${PBC}"
time "${BINARY}" "${H}" "${PBC}"

echo ""
echo "[DONE] data/h_null/observables/OBC/gap_obc_L*.dat  obs_obc_L*.dat"
