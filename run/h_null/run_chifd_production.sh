#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

DH="5e-4"
OUT_DIR="${ROOT_DIR}/data/h_null/chiz_fd/dh_5e-04"
PBC_OUT_DIR="${OUT_DIR}/PBC"
OBC_OUT_DIR="${OUT_DIR}/OBC"
OBS_DIR="${ROOT_DIR}/data/h_null/observables"
ED_L=(4 6 8 10 12)
LZ_L=(14 16 18 20)
ALL_L=(4 6 8 10 12 14 16 18 20)
BCS=("PBC" "OBC")
DRY_RUN=0

nproc_default() {
    if command -v nproc >/dev/null 2>&1; then
        nproc
    else
        echo 1
    fi
}

NPROC="$(nproc_default)"
if (( NPROC < 1 )); then
    NPROC=1
fi
if (( NPROC < 4 )); then
    JOBS_ED="$NPROC"
else
    JOBS_ED=4
fi
JOBS_LZ=1

usage() {
    cat <<USAGE
Usage: $0 [--jobs-ed N] [--jobs-lz N] [--dry-run]

Runs production chi_z finite-difference data for:
  ED      L = 4,6,8,10,12
  Lanczos L = 14,16,18,20

dh is fixed to ${DH}. Smoke options are never used in production.
Partial valid outputs are left in place; ising_chiz_fd resumes them automatically.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --jobs-ed)
            [[ $# -ge 2 ]] || { echo "ERROR: --jobs-ed requires N" >&2; exit 2; }
            JOBS_ED="$2"
            shift 2
            ;;
        --jobs-lz)
            [[ $# -ge 2 ]] || { echo "ERROR: --jobs-lz requires N" >&2; exit 2; }
            JOBS_LZ="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$JOBS_ED" in (*[!0-9]*|"") echo "ERROR: --jobs-ed must be a positive integer" >&2; exit 2;; esac
case "$JOBS_LZ" in (*[!0-9]*|"") echo "ERROR: --jobs-lz must be a positive integer" >&2; exit 2;; esac
if (( JOBS_ED < 1 || JOBS_LZ < 1 )); then
    echo "ERROR: jobs must be positive" >&2
    exit 2
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT_DIR}/logs/chiz_fd/${timestamp}"

method_code_for_L() {
    local L="$1"
    if (( L <= 12 )); then
        echo 0
    else
        echo 1
    fi
}

pbc_arg_for_bc() {
    local BC="$1"
    if [[ "$BC" == "PBC" ]]; then
        echo 1
    else
        echo 0
    fi
}

output_file_for() {
    local L="$1" BC="$2"
    if [[ "$BC" == "PBC" ]]; then
        printf "%s/chizfd_L%02d.dat" "$PBC_OUT_DIR" "$L"
    else
        printf "%s/chizfd_obc_L%02d.dat" "$OBC_OUT_DIR" "$L"
    fi
}

source_file_for() {
    local L="$1" BC="$2"
    if (( L <= 12 )); then
        if [[ "$BC" == "PBC" ]]; then
            printf "%s/PBC/gap_L%02d.dat" "$OBS_DIR" "$L"
        else
            printf "%s/OBC/gap_obc_L%02d.dat" "$OBS_DIR" "$L"
        fi
    else
        if [[ "$BC" == "PBC" ]]; then
            printf "%s/PBC/gap_lz_L%02d.dat" "$OBS_DIR" "$L"
        else
            printf "%s/OBC/gap_lz_obc_L%02d.dat" "$OBS_DIR" "$L"
        fi
    fi
}

log_file_for() {
    local L="$1" BC="$2"
    printf "%s/L%02d_%s.log" "$LOG_DIR" "$L" "$BC"
}

quoted_cmd_for() {
    local L="$1" BC="$2" log="$3" pbc
    pbc="$(pbc_arg_for_bc "$BC")"
    printf '/usr/bin/time -p "%s/ising_chiz_fd" %s %s %s > "%s" 2>&1' \
        "$ROOT_DIR" "$pbc" "$DH" "$L" "$log"
}

VALIDATION_SUMMARY=""
validate_output() {
    local L="$1" BC="$2"
    local file source method
    file="$(output_file_for "$L" "$BC")"
    source="$(source_file_for "$L" "$BC")"
    method="$(method_code_for_L "$L")"

    if [[ ! -f "$file" ]]; then
        VALIDATION_SUMMARY="missing output file: $file"
        return 1
    fi
    if [[ ! -f "$source" ]]; then
        VALIDATION_SUMMARY="missing static source file: $source"
        return 1
    fi
    if grep -q '^# smoke_\(max_g_points\|g_window\)' "$file"; then
        VALIDATION_SUMMARY="smoke header present"
        return 1
    fi

    local out
    if out="$(awk -v method="$method" '
        FNR == NR {
            if ($0 !~ /^#/ && NF > 0) {
                nsrc++;
                gsrc[nsrc] = $1 + 0.0;
            }
            next;
        }
        $0 ~ /^#/ || NF == 0 { next; }
        {
            n++;
            if (NF != 10) bad_nf++;
            g = $1 + 0.0;
            mc = $3 + 0;
            chi = $8 + 0.0;
            o1 = $9 + 0.0;
            o2 = $10 + 0.0;
            if (mc != method) bad_method++;
            if (chi <= 0.0) bad_chi++;
            d = g - gsrc[n];
            if (d < 0.0) d = -d;
            if (d > max_gdiff) max_gdiff = d;
            if (o1 > max_odd1) max_odd1 = o1;
            if (o2 > max_odd2) max_odd2 = o2;
        }
        END {
            printf "rows=%d source_rows=%d max_gdiff=%.3e method=%d bad_method=%d bad_chi=%d bad_nf=%d max_oddness1=%.12e max_oddness2=%.12e",
                   n, nsrc, max_gdiff, method, bad_method, bad_chi, bad_nf, max_odd1, max_odd2;
            if (n != nsrc) exit 11;
            if (max_gdiff >= 1e-12) exit 12;
            if (bad_method > 0) exit 13;
            if (bad_chi > 0) exit 14;
            if (bad_nf > 0) exit 15;
        }
    ' "$source" "$file")"; then
        VALIDATION_SUMMARY="$out"
        return 0
    else
        VALIDATION_SUMMARY="$out"
        return 1
    fi
}

RUN_LIST=()
SKIP_LIST=()
RECALC_LIST=()

classify_one() {
    local L="$1" BC="$2" phase="$3"
    local file source pbc log
    file="$(output_file_for "$L" "$BC")"
    source="$(source_file_for "$L" "$BC")"
    pbc="$(pbc_arg_for_bc "$BC")"
    log="$(log_file_for "$L" "$BC")"

    if validate_output "$L" "$BC"; then
        SKIP_LIST+=("${phase} L=${L} BC=${BC} file=${file} (${VALIDATION_SUMMARY})")
    else
        RECALC_LIST+=("${phase} L=${L} BC=${BC} file=${file} action=run-or-resume reason=${VALIDATION_SUMMARY}")
        RUN_LIST+=("${L}|${BC}|${phase}|${log}")
    fi
}

for L in "${ED_L[@]}"; do
    for BC in "${BCS[@]}"; do
        classify_one "$L" "$BC" "ED"
    done
done
for L in "${LZ_L[@]}"; do
    for BC in "${BCS[@]}"; do
        classify_one "$L" "$BC" "Lanczos"
    done
done

echo "====================================================="
echo "chi_z finite-difference production"
echo "====================================================="
echo "dh           = ${DH}"
echo "jobs-ed      = ${JOBS_ED}"
echo "jobs-lz      = ${JOBS_LZ}"
echo "dry-run      = ${DRY_RUN}"
echo "log_dir      = ${LOG_DIR}"
echo "threads      = OMP=${OMP_NUM_THREADS} OPENBLAS=${OPENBLAS_NUM_THREADS} MKL=${MKL_NUM_THREADS} BLIS=${BLIS_NUM_THREADS}"
echo

echo "Already complete / skipped:"
if (( ${#SKIP_LIST[@]} == 0 )); then
    echo "  (none)"
else
    printf "  %s\n" "${SKIP_LIST[@]}"
fi
echo

echo "Would run/resume / scheduled:"
if (( ${#RECALC_LIST[@]} == 0 )); then
    echo "  (none)"
else
    printf "  %s\n" "${RECALC_LIST[@]}"
fi
echo

echo "Commands:"
if (( ${#RUN_LIST[@]} == 0 )); then
    echo "  (none)"
else
    for show_phase in "ED" "Lanczos"; do
        echo "  Phase ${show_phase}:"
        printed=0
        for item in "${RUN_LIST[@]}"; do
            IFS='|' read -r L BC phase log <<< "$item"
            [[ "$phase" == "$show_phase" ]] || continue
            printf "    %s\n" "$(quoted_cmd_for "$L" "$BC" "$log")"
            printed=1
        done
        if (( printed == 0 )); then
            echo "    (none)"
        fi
    done
fi
echo

if (( DRY_RUN )); then
    echo "Dry-run only: no production command launched."
    exit 0
fi

mkdir -p "$LOG_DIR" "$OUT_DIR" "$PBC_OUT_DIR" "$OBC_OUT_DIR" "${ROOT_DIR}/plots/debug"

run_one() {
    local L="$1" BC="$2" phase="$3" log="$4" pbc
    local cmd
    pbc="$(pbc_arg_for_bc "$BC")"
    cmd=(/usr/bin/time -p "${ROOT_DIR}/ising_chiz_fd" "$pbc" "$DH" "$L")
    echo "START ${phase} L=${L} BC=${BC} log=${log} (resume-capable)"
    if ! "${cmd[@]}" >"$log" 2>&1; then
        echo "ERROR: run failed for L=${L} BC=${BC}; see ${log}" >&2
        return 1
    fi
    if ! validate_output "$L" "$BC"; then
        echo "ERROR: validation failed after run for L=${L} BC=${BC}: ${VALIDATION_SUMMARY}" >&2
        echo "       see ${log}" >&2
        return 1
    fi
    echo "DONE ${phase} L=${L} BC=${BC}: ${VALIDATION_SUMMARY}"
}

run_phase() {
    local phase="$1" jobs="$2"
    local active=0 item L BC item_phase log

    echo "-----------------------------------------------------"
    echo "Phase ${phase}, jobs=${jobs}"
    echo "-----------------------------------------------------"

    for item in "${RUN_LIST[@]}"; do
        IFS='|' read -r L BC item_phase log <<< "$item"
        [[ "$item_phase" == "$phase" ]] || continue
        run_one "$L" "$BC" "$phase" "$log" &
        active=$((active + 1))
        if (( active >= jobs )); then
            wait -n
            active=$((active - 1))
        fi
    done
    while (( active > 0 )); do
        wait -n
        active=$((active - 1))
    done
}

run_phase "ED" "$JOBS_ED"
run_phase "Lanczos" "$JOBS_LZ"

echo "-----------------------------------------------------"
echo "Final validation for all official sizes"
echo "-----------------------------------------------------"
for L in "${ALL_L[@]}"; do
    for BC in "${BCS[@]}"; do
        if ! validate_output "$L" "$BC"; then
            echo "ERROR: final validation failed for L=${L} BC=${BC}: ${VALIDATION_SUMMARY}" >&2
            exit 1
        fi
        echo "PASS L=${L} BC=${BC}: ${VALIDATION_SUMMARY}"
    done
done

python3 "${ROOT_DIR}/scripts/h_null/check_chiz_grid.py" --dh "$DH"

plot_script="${ROOT_DIR}/plots/debug/chiz_fd_ALL_L.gnuplot"
cat > "$plot_script" <<GNUPLOT
set terminal pngcairo enhanced size 1800,800 font "Arial,10"
set datafile commentschars "#"
set grid
set key outside right
set xrange [0.4:1.6]
set xlabel "g"
set ylabel "{/Symbol c}_z^{FD}"
set arrow 1 from 1.0, graph 0 to 1.0, graph 1 nohead dt 2 lc rgb "#555555" lw 1.4 front

set output "${ROOT_DIR}/plots/debug/chiz_fd_ALL_L_logy.png"
set logscale y
set multiplot layout 1,2 title "chi_z finite-difference production"
set title "PBC"
plot \
 "${PBC_OUT_DIR}/chizfd_L04.dat" using 1:8 with lines title "L=4", \
 "${PBC_OUT_DIR}/chizfd_L06.dat" using 1:8 with lines title "L=6", \
 "${PBC_OUT_DIR}/chizfd_L08.dat" using 1:8 with lines title "L=8", \
 "${PBC_OUT_DIR}/chizfd_L10.dat" using 1:8 with lines title "L=10", \
 "${PBC_OUT_DIR}/chizfd_L12.dat" using 1:8 with lines title "L=12", \
 "${PBC_OUT_DIR}/chizfd_L14.dat" using 1:8 with lines title "L=14", \
 "${PBC_OUT_DIR}/chizfd_L16.dat" using 1:8 with lines title "L=16", \
 "${PBC_OUT_DIR}/chizfd_L18.dat" using 1:8 with lines title "L=18", \
 "${PBC_OUT_DIR}/chizfd_L20.dat" using 1:8 with lines title "L=20"
set title "OBC"
plot \
 "${OBC_OUT_DIR}/chizfd_obc_L04.dat" using 1:8 with lines title "L=4", \
 "${OBC_OUT_DIR}/chizfd_obc_L06.dat" using 1:8 with lines title "L=6", \
 "${OBC_OUT_DIR}/chizfd_obc_L08.dat" using 1:8 with lines title "L=8", \
 "${OBC_OUT_DIR}/chizfd_obc_L10.dat" using 1:8 with lines title "L=10", \
 "${OBC_OUT_DIR}/chizfd_obc_L12.dat" using 1:8 with lines title "L=12", \
 "${OBC_OUT_DIR}/chizfd_obc_L14.dat" using 1:8 with lines title "L=14", \
 "${OBC_OUT_DIR}/chizfd_obc_L16.dat" using 1:8 with lines title "L=16", \
 "${OBC_OUT_DIR}/chizfd_obc_L18.dat" using 1:8 with lines title "L=18", \
 "${OBC_OUT_DIR}/chizfd_obc_L20.dat" using 1:8 with lines title "L=20"
unset multiplot

unset logscale y
set output "${ROOT_DIR}/plots/debug/chiz_fd_ALL_L_linear.png"
set multiplot layout 1,2 title "chi_z finite-difference production"
set title "PBC"
plot \
 "${PBC_OUT_DIR}/chizfd_L04.dat" using 1:8 with lines title "L=4", \
 "${PBC_OUT_DIR}/chizfd_L06.dat" using 1:8 with lines title "L=6", \
 "${PBC_OUT_DIR}/chizfd_L08.dat" using 1:8 with lines title "L=8", \
 "${PBC_OUT_DIR}/chizfd_L10.dat" using 1:8 with lines title "L=10", \
 "${PBC_OUT_DIR}/chizfd_L12.dat" using 1:8 with lines title "L=12", \
 "${PBC_OUT_DIR}/chizfd_L14.dat" using 1:8 with lines title "L=14", \
 "${PBC_OUT_DIR}/chizfd_L16.dat" using 1:8 with lines title "L=16", \
 "${PBC_OUT_DIR}/chizfd_L18.dat" using 1:8 with lines title "L=18", \
 "${PBC_OUT_DIR}/chizfd_L20.dat" using 1:8 with lines title "L=20"
set title "OBC"
plot \
 "${OBC_OUT_DIR}/chizfd_obc_L04.dat" using 1:8 with lines title "L=4", \
 "${OBC_OUT_DIR}/chizfd_obc_L06.dat" using 1:8 with lines title "L=6", \
 "${OBC_OUT_DIR}/chizfd_obc_L08.dat" using 1:8 with lines title "L=8", \
 "${OBC_OUT_DIR}/chizfd_obc_L10.dat" using 1:8 with lines title "L=10", \
 "${OBC_OUT_DIR}/chizfd_obc_L12.dat" using 1:8 with lines title "L=12", \
 "${OBC_OUT_DIR}/chizfd_obc_L14.dat" using 1:8 with lines title "L=14", \
 "${OBC_OUT_DIR}/chizfd_obc_L16.dat" using 1:8 with lines title "L=16", \
 "${OBC_OUT_DIR}/chizfd_obc_L18.dat" using 1:8 with lines title "L=18", \
 "${OBC_OUT_DIR}/chizfd_obc_L20.dat" using 1:8 with lines title "L=20"
unset multiplot
unset output
GNUPLOT

gnuplot "$plot_script"

echo "====================================================="
echo "Production complete. Logs: ${LOG_DIR}"
echo "Plots:"
echo "  ${ROOT_DIR}/plots/debug/chiz_fd_ALL_L_logy.png"
echo "  ${ROOT_DIR}/plots/debug/chiz_fd_ALL_L_linear.png"
echo "  ${ROOT_DIR}/plots/debug/chiz_fd_ALL_L.gnuplot"
