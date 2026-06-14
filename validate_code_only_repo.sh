#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$PWD}"
cd "$ROOT"

echo "Validating code-only repository: $PWD"
echo "File count: $(find . -type f | wc -l)"

fail=0

for forbidden in \
  '.venv' \
  '__pycache__' \
  'logs' \
  'obj' \
  'trash_review' \
  '.work' \
  '.locks' \
  'backups' \
  'snapshot'; do
  if find . \( -name "$forbidden" -o -path "*/$forbidden/*" \) -print -quit | grep -q .; then
    echo "Forbidden path present: $forbidden" >&2
    fail=1
  fi
done

# No generated numerical data or final plots should be versioned.
if find data -type f ! -name ".gitkeep" ! -name "README.md" -print -quit | grep -q .; then
  echo "Generated data files are still present under data/" >&2
  find data -type f ! -name ".gitkeep" ! -name "README.md" | sed -n '1,40p' >&2
  fail=1
fi

if find data -type d \( -name "matplotlib_cache" -o -name "work" \) -print -quit | grep -q .; then
  echo "Generated cache/work directories are still present under data/" >&2
  find data -type d \( -name "matplotlib_cache" -o -name "work" \) | sed -n '1,40p' >&2
  fail=1
fi

if find plots -type f ! -path "plots/readme/*" ! -name ".gitkeep" ! -name "README.md" -print -quit | grep -q .; then
  echo "Generated plot/media files are still present under plots/" >&2
  find plots -type f ! -path "plots/readme/*" ! -name ".gitkeep" ! -name "README.md" | sed -n '1,40p' >&2
  fail=1
fi

required=(
  "Makefile"
  "README.md"
  ".gitignore"
  "docs/reproducibility.md"
  "docs/pipeline.md"
  "src/main_quench.c"
  "src/spectral/main_spectral.c"
  "src/h_null/main_static.c"
  "src/h_null/main_lanczos.c"
  "src/h_null/main_chiz_fd.c"
  "src/main_hfield.c"
  "src/h_field/gh_surface/gh_surface.c"
  "include/hamiltonian.h"
  "include/lanczos.h"
  "scripts/h_null/fss_h_null_analysis.py"
  "scripts/h_null/plot_observables.py"
  "scripts/h_field/plot_hfield.py"
  "scripts/h_field/compute/run_hfield_grid.py"
  "scripts/h_field/gh_surface/run_gh_surface.py"
  "scripts/quench/plot_quench.py"
  "scripts/quench/export_quench_constants.py"
  "scripts/quench/run_quench_ed.sh"
  "scripts/spectral/plot_spectral.py"
  "data/README.md"
  "plots/README.md"
)

for path in "${required[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "Required path missing: $path" >&2
    fail=1
  fi
done

# The local-only --use-local-extensions flag is allowed; public extension
# generators, binaries, data skeletons and source trees are not.
for forbidden_path in \
  "scripts/server" \
  "scripts/analysis" \
  "scripts/run" \
  "run/run_quench.sh" \
  "ising_gap_ext" \
  "ising_mz_ext" \
  "src/h_field/gap_ext" \
  "src/h_field/mz_ext" \
  "scripts/h_field/compute/run_gap_extension.py" \
  "scripts/h_field/compute/run_mz_extension.py" \
  "scripts/h_field/plot_gap_extension.py" \
  "scripts/h_field/plot_mz_extension.py" \
  "data/h_field/gap_ext" \
  "data/h_field/mz_ext" \
  "docs/cleanup_inventory" \
  "scripts/spectral/compute" \
  "scripts/spectral/compute/run_spectral_grid.py" \
  "scripts/spectral/compute/run_spectral_report_distributions_final.py" \
  "scripts/spectral/compute/run_spectral_report_r_grid.py" \
  "scripts/spectral/report_spectral_coverage.py" \
  "scripts/spectral/spectral_report_config.py" \
  "scripts/spectral/plot_spectral_report.py"; do
  if [[ -e "$forbidden_path" ]]; then
    echo "Forbidden public-scope path present: $forbidden_path" >&2
    fail=1
  fi
done

if [[ "$fail" -ne 0 ]]; then
  echo "Validation FAILED" >&2
  exit 1
fi

echo "Validation OK"
