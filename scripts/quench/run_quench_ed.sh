#!/usr/bin/env bash
set -euo pipefail

# By default this driver imports the critical scaling constants from the h=0
# FSS pipeline. Use USE_FSS_CONSTANTS=0 to run with manual/theoretical values.
# The imported values are audited in data/quench/quench_constants_used.json.

cd "$(dirname "$0")/../.."

BIN="./ising_quench"
LOG_DIR="logs/quench"
LOG_FILE="$LOG_DIR/run_quench_ed.log"

mkdir -p "$LOG_DIR" data/quench/cqt data/quench/foqt data/quench/loschmidt plots/quench

if [[ ! -x "$BIN" ]]; then
    echo "[ERROR] $BIN not found. Run: make quench" >&2
    exit 1
fi

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

L_VALUES=(4 6 8 10 12)
BC_VALUES=(pbc obc)
G_VALUES=(0.500 0.900)
PHI_VALUES=(1.000 3.000)

BETA_OVER_NU="${BETA_OVER_NU:-0.125}"
NU="${NU:-1.0}"
Y_H="${Y_H:-1.875}"
G_PC="${G_PC:-1.0}"
Z="${Z:-1.0}"
THETA_MAX="${THETA_MAX:-10.0}"
NTHETA_CQT="${NTHETA_CQT:-1001}"
NTHETA_FOQT="${NTHETA_FOQT:-2001}"
THETA_MAX_LOSCH="${THETA_MAX_LOSCH:-7.0}"
NTHETA_LOSCH="${NTHETA_LOSCH:-1401}"
KAPPA0="${KAPPA0:-1.0}"
KAPPA1="${KAPPA1:--1.0}"
RUN_LOSCHMIDT="${RUN_LOSCHMIDT:-1}"
USE_FSS_CONSTANTS="${USE_FSS_CONSTANTS:-1}"
FSS_CONSTANTS_FILE="${FSS_CONSTANTS_FILE:-data/h_null/fss/fss_constants.json}"

log() {
    printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "$LOG_FILE"
}

run_job() {
    local label="$1"
    shift
    log "START $label"
    "$@" 2> >(while IFS= read -r line; do log "WARN/ERR $label: $line"; done)
    log "DONE  $label"
}

: > "$LOG_FILE"
log "Quench ED production starts"
log "theta_max=$THETA_MAX NTHETA_CQT=$NTHETA_CQT NTHETA_FOQT=$NTHETA_FOQT THETA_MAX_LOSCH=$THETA_MAX_LOSCH NTHETA_LOSCH=$NTHETA_LOSCH OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"

if [[ "$USE_FSS_CONSTANTS" == "1" ]]; then
    eval "$(python3 scripts/quench/export_quench_constants.py --input "$FSS_CONSTANTS_FILE")"
    log "Using h=0 FSS constants from $QUENCH_CONSTANTS_SOURCE"
else
    log "[WARN] Using theoretical/default critical exponents, not h=0 FSS constants."
    PBC_GPC="${PBC_GPC:-$G_PC}"
    PBC_BETA_OVER_NU="${PBC_BETA_OVER_NU:-$BETA_OVER_NU}"
    PBC_NU="${PBC_NU:-$NU}"
    PBC_Z="${PBC_Z:-$Z}"
    PBC_Y_H="${PBC_Y_H:-$Y_H}"
    OBC_GPC="${OBC_GPC:-$G_PC}"
    OBC_BETA_OVER_NU="${OBC_BETA_OVER_NU:-$BETA_OVER_NU}"
    OBC_NU="${OBC_NU:-$NU}"
    OBC_Z="${OBC_Z:-$Z}"
    OBC_Y_H="${OBC_Y_H:-$Y_H}"
    QUENCH_CONSTANTS_SOURCE="manual/default"
fi

log "PBC constants: g_pc=$PBC_GPC beta_over_nu=$PBC_BETA_OVER_NU nu=$PBC_NU z=$PBC_Z y_h=$PBC_Y_H"
log "OBC constants: g_pc=$OBC_GPC beta_over_nu=$OBC_BETA_OVER_NU nu=$OBC_NU z=$OBC_Z y_h=$OBC_Y_H"

log "CQT campaign"
for bc in "${BC_VALUES[@]}"; do
    for L in "${L_VALUES[@]}"; do
        if [[ "$bc" == "pbc" ]]; then
            GPC="$PBC_GPC"
            BOVNU="$PBC_BETA_OVER_NU"
            NU_BC="$PBC_NU"
            Z_BC="$PBC_Z"
            YH_BC="$PBC_Y_H"
        else
            GPC="$OBC_GPC"
            BOVNU="$OBC_BETA_OVER_NU"
            NU_BC="$OBC_NU"
            Z_BC="$OBC_Z"
            YH_BC="$OBC_Y_H"
        fi
        run_job "CQT $bc L=$L" \
            "$BIN" cqt "$L" "$bc" "$GPC" "$BOVNU" "$NU_BC" "$Z_BC" "$YH_BC" "$THETA_MAX" "$NTHETA_CQT"
    done
done

log "FOQT campaign"
log "FOQT campaign does not use h=0 CQT exponents; it uses Delta0(g,L) and m0(g)."
for bc in "${BC_VALUES[@]}"; do
    for g in "${G_VALUES[@]}"; do
        for L in "${L_VALUES[@]}"; do
            run_job "FOQT $bc g=$g L=$L" \
                "$BIN" foqt "$L" "$bc" "$g" "$KAPPA0" "$KAPPA1" "$THETA_MAX" "$NTHETA_FOQT"
        done
    done
done

if [[ "$RUN_LOSCHMIDT" != "0" ]]; then
    log "Loschmidt campaign"
    for bc in "${BC_VALUES[@]}"; do
        if [[ "$bc" == "pbc" ]]; then
            GPC="$PBC_GPC"
            Z_BC="$PBC_Z"
            YH_BC="$PBC_Y_H"
        else
            GPC="$OBC_GPC"
            Z_BC="$OBC_Z"
            YH_BC="$OBC_Y_H"
        fi
        for Phi in "${PHI_VALUES[@]}"; do
            for L in "${L_VALUES[@]}"; do
                run_job "LOSCH $bc Phi=$Phi L=$L" \
                    "$BIN" loschmidt "$L" "$bc" "$GPC" "$Phi" "$YH_BC" "$Z_BC" "$THETA_MAX_LOSCH" "$NTHETA_LOSCH"
            done
        done
    done
else
    log "Loschmidt campaign skipped because RUN_LOSCHMIDT=0"
fi

log "Quench ED production complete"
