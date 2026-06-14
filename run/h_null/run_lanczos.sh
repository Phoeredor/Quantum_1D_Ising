#!/usr/bin/env bash
# =============================================================================
# run_lanczos.sh - Production runs for Lanczos-based FSS gap + observable analysis.
#
# Usage:
#   bash run/h_null/run_lanczos.sh [--resume] [--campaign {1|2|all}]
#
# Campaign 1: PBC gap + obs vs g, L=14,16,18,20,22  (FSS near CQT)
# Campaign 2: OBC gap + obs vs g, L=14,16,18,20,22  (exponential gap check)
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/../.."

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BIN=./ising_lanczos
OUTDIR=data/h_null/observables
LOGFILE=data/h_null/observables/run_lanczos.log

# Campaign 1: PBC, FSS near CQT (adaptive g-grid built inside the binary)
C1_L_VALUES=(14 16 18 20 22)
C1_H=0.0
C1_PBC=1

# Campaign 2: OBC, ordered phase (adaptive g-grid built inside the binary)
C2_L_VALUES=(14 16 18 20 22)
C2_H=0.0
C2_PBC=0

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

RESUME=0
CAMPAIGN=all

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)   RESUME=1;      shift ;;
        --campaign) CAMPAIGN="$2"; shift 2 ;;
        *)          echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

mkdir -p "$OUTDIR/PBC" "$OUTDIR/OBC" "$(dirname "$LOGFILE")"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOGFILE"; }

cleanup() {
    log "INTERRUPTED — no partial files to clean (Lanczos writes atomically)"
    exit 1
}
trap cleanup SIGINT SIGTERM

ERRORS=0

if [[ ! -x "$BIN" ]]; then
    echo "[ERROR] $BIN not found or not executable. Run: make lanczos"
    exit 1
fi

log "======================================================="
log "run_lanczos.sh  campaign=$CAMPAIGN  resume=$RESUME"
log "======================================================="

# ---------------------------------------------------------------------------
# Helper: run one Lanczos job for a given L
# ---------------------------------------------------------------------------
# run_one L h PBC

run_one() {
    local L="$1" H="$2" PBC="$3"

    # Expected output filenames (must match write_gap/obs_file() in main_lanczos.c)
    local suffix=""
    local bc_dir="$OUTDIR/PBC"
    if [[ "$PBC" -eq 0 ]]; then
        suffix="_obc"
        bc_dir="$OUTDIR/OBC"
    fi
    local out_file
    out_file=$(printf "%s/gap_lz%s_L%02d.dat" "$bc_dir" "$suffix" "$L")
    local obs_file
    obs_file=$(printf "%s/obs_lz%s_L%02d.dat" "$bc_dir" "$suffix" "$L")

    if [[ $RESUME -eq 1 ]] && [[ -s "$out_file" ]] && [[ -s "$obs_file" ]]; then
        log "SKIP (exists): $out_file  $obs_file"
        return 0
    fi

    "$BIN" "$H" "$PBC" "$L"
}

# ---------------------------------------------------------------------------
# Parallel helpers
# ---------------------------------------------------------------------------

declare -a JOB_PIDS=()
declare -a JOB_LABELS=()

launch_job() {
    local L="$1" H="$2" PBC="$3" BC_TAG="$4"

    (
        log "START L=$L $BC_TAG"
        t0=$(date +%s)

        status=0
        if run_one "$L" "$H" "$PBC"; then
            status=0
        else
            status=$?
        fi

        elapsed=$(( $(date +%s) - t0 ))
        if [[ $status -eq 0 ]]; then
            log "DONE  L=$L  ${elapsed}s"
        else
            log "[ERROR] L=$L  exit=$status  ${elapsed}s"
        fi

        exit $status
    ) &

    JOB_PIDS+=("$!")
    JOB_LABELS+=("L=$L $BC_TAG")
}

wait_jobs() {
    local i
    local status
    for i in "${!JOB_PIDS[@]}"; do
        if wait "${JOB_PIDS[$i]}"; then
            :
        else
            status=$?
            ERRORS=$(( ERRORS + 1 ))
            log "[ERROR] job failed: ${JOB_LABELS[$i]} (exit=$status)"
        fi
    done
    JOB_PIDS=()
    JOB_LABELS=()
}

# ---------------------------------------------------------------------------
# Campaign execution
# ---------------------------------------------------------------------------

if [[ "$CAMPAIGN" == "all" ]]; then
    log "--- CAMPAIGN all: phased PBC+OBC parallel ---"
    export OPENBLAS_NUM_THREADS=1
    export OMP_NUM_THREADS=1

    log "--- Phase A1: PBC+OBC for L=14,16 ---"
    for L in 14 16; do
        launch_job "$L" "$C1_H" "$C1_PBC" "PBC"
        launch_job "$L" "$C2_H" "$C2_PBC" "OBC"
    done
    wait_jobs

    log "--- Phase A2: PBC+OBC for L=18,20 ---"
    for L in 18 20; do
        launch_job "$L" "$C1_H" "$C1_PBC" "PBC"
        launch_job "$L" "$C2_H" "$C2_PBC" "OBC"
    done
    wait_jobs

    log "--- Phase B: PBC+OBC for L=22 ---"
    launch_job "22" "$C1_H" "$C1_PBC" "PBC"
    launch_job "22" "$C2_H" "$C2_PBC" "OBC"
    wait_jobs

    log "--- CAMPAIGN all complete ---"

elif [[ "$CAMPAIGN" == "1" ]]; then
    log "--- CAMPAIGN 1: PBC FSS, parallel ---"
    export OPENBLAS_NUM_THREADS=1
    export OMP_NUM_THREADS=1

    for L in "${C1_L_VALUES[@]}"; do
        launch_job "$L" "$C1_H" "$C1_PBC" "PBC"
    done
    wait_jobs

    log "--- CAMPAIGN 1 complete ---"

elif [[ "$CAMPAIGN" == "2" ]]; then
    log "--- CAMPAIGN 2: OBC ordered, parallel ---"
    export OPENBLAS_NUM_THREADS=1
    export OMP_NUM_THREADS=1

    for L in "${C2_L_VALUES[@]}"; do
        launch_job "$L" "$C2_H" "$C2_PBC" "OBC"
    done
    wait_jobs

    log "--- CAMPAIGN 2 complete ---"

else
    echo "[ERROR] invalid --campaign value: $CAMPAIGN (expected 1,2,all)"
    exit 1
fi

log "======================================================="
if [[ $ERRORS -gt 0 ]]; then
    log "WARNING: $ERRORS run(s) failed — check log above"
    echo "[run_lanczos.sh] $ERRORS error(s) — see $LOGFILE" >&2
fi
log "All campaigns done."
log "Run: python3 scripts/h_null/plot_observables.py --lanczos"
log "======================================================="

exit "$ERRORS"
