# Public Pipeline

This file lists the supported public entrypoints. Commands generate data and
plots locally; generated outputs are intentionally ignored by Git.

## Build

```bash
make all
```

Equivalent public targets:

```bash
make static
make quench
make lanczos
make spectral
make hfield
make ghsurface
make chizfd
```

Use `make clean` to remove C objects and executables.

## h=0

Generate full ED and Lanczos h=0 observables:

```bash
bash run/h_null/run_pbc.sh
bash run/h_null/run_obc.sh
bash run/h_null/run_lanczos.sh --resume --campaign all
```

Generate finite-difference susceptibility data:

```bash
bash run/h_null/run_chifd_production.sh
```

Check and plot h=0 outputs:

```bash
python3 scripts/h_null/check_data_h_null.py
python3 scripts/h_null/plot_observables.py --lanczos
python3 scripts/h_null/fss_h_null_analysis.py --lanczos
```

The FSS script writes constants consumed by the h-field and quench pipelines.

## h-field

Build the executable and generate CQT/FOQT h-field grids:

```bash
make hfield
python3 scripts/h_field/compute/run_hfield_grid.py --resume
```

Plot the public h-field figures:

```bash
python3 scripts/h_field/plot_hfield.py
```

The public plotter reads `data/h_field/hfield_raw/` and writes derived tables
to `data/h_field/hfield_processed/` and figures to `plots/hfield/`.

Local extension data are disabled by default and are not part of the public
pipeline. Advanced local runs may pass `--use-local-extensions` to let
`plot_hfield.py` look for local-only `mz_ext`/`gap_ext` data if present. This
flag is not needed to reproduce the public h-field pipeline.

## gh-surface

Build and run the two-parameter surface worker:

```bash
make ghsurface
python3 scripts/h_field/gh_surface/run_gh_surface.py --production-confirm --resume
```

Plot surface outputs:

```bash
python3 scripts/h_field/gh_surface/plot_gh_surface.py
python3 scripts/h_field/gh_surface/plot_gh_surface_collapse.py
```

Data are written under `data/h_field/gh_surface/`.

## Quench

Build the ED quench executable and run the public campaign driver:

```bash
make quench
bash scripts/quench/run_quench_ed.sh
```

The driver covers CQT, FOQT and Loschmidt ED quench datasets and imports h=0
FSS constants by default.

Plot quench outputs:

```bash
python3 scripts/quench/plot_quench.py
```

## Spectral

Build the ED-only spectral worker:

```bash
make spectral
```

Single realization example:

```bash
./ising_spectral --L 12 --g 1.0 --h 0.0 --omega 0.3 \
  --realization 0 --master-seed 12345 \
  --out data/spectral/raw/example.dat --overwrite
```

Generated raw data may come from `ising_spectral` directly or from local user
campaigns outside the public repository. The public Python surface is limited
to plotting already generated raw data:

```bash
python3 scripts/spectral/plot_spectral.py
```

The plotter reads `data/spectral/raw/` and, if present,
`data/spectral/raw/bulk0p5/`. It writes figures under `plots/spectral/report/`
and processed tables under `data/spectral/processed/report/`.
