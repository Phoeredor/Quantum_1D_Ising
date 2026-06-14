#!/usr/bin/env bash
# run_pbc.sh — PBC sweep, h=0

set -euo pipefail
cd "$(dirname "$0")/../.."

BINARY="./ising_static"
H="0.0"
PBC="1"

echo "=============================================="
echo "  PBC production run"
echo "  g-grid: adaptive [0.4, 1.6], gc=1.0, nu=1.0"
echo "  L: 4 6 8 10 12  (hardcoded in src/h_null/main_static.c)"
echo "=============================================="

[[ -x "${BINARY}" ]] || { echo "[ERROR] '${BINARY}' not found. Run 'make' first."; exit 1; }

mkdir -p data/h_null/observables/PBC

echo "[RUN] ${BINARY} ${H} ${PBC}"
time "${BINARY}" "${H}" "${PBC}"

echo ""
echo "[DONE] data/h_null/observables/PBC/gap_L*.dat  obs_L*.dat"
