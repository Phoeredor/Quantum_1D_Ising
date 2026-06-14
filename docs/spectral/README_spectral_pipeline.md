# Spectral Plotting

The public repository keeps the spectral section minimal:

- C source: `src/spectral/main_spectral.c`
- public Python script: `scripts/spectral/plot_spectral.py`
- raw-data skeleton: `data/spectral/raw/`
- processed-output skeleton: `data/spectral/processed/`
- plot skeleton: `plots/spectral/`

The Python script plots already generated ED spectral-statistics data. Raw
data may be produced directly with `ising_spectral` or by local user campaigns
outside the public repository.

Supported plot outputs:

- `P(s)`
- `P(r)`
- `<r>(h)`

Input locations:

- `data/spectral/raw/`
- `data/spectral/raw/bulk0p5/`, if present

Output locations:

- `plots/spectral/report/`
- `data/spectral/processed/report/`

No spectral production, coverage, debug, wrapper, or report-driver scripts are
part of the public `scripts/spectral/` surface.
