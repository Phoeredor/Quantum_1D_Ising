# Quantum 1D Ising: Exact Diagonalization of the 1D Quantum Ising Chain

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Linux](https://img.shields.io/badge/Linux-FCC624.svg?style=flat&logo=linux&logoColor=black)
![C](https://img.shields.io/badge/C99-00599C.svg?style=flat&logo=c&logoColor=white)
![Python](https://img.shields.io/badge/Python-%E2%89%A5%203.8-3776AB.svg?style=flat&logo=python&logoColor=white)
![GCC](https://img.shields.io/badge/GCC-00599C.svg?style=flat&logo=gnu&logoColor=white)
![LAPACK/OpenBLAS](https://img.shields.io/badge/LAPACK%2FOpenBLAS-required-orange)

**Module 2 — Numerical Methods for Physics, University of Pisa**

Developed as a course project for the ED/FSS module taught by Prof. Davide Rossini.

Exact diagonalization workflows for the one-dimensional quantum Ising chain with transverse field $g$ and longitudinal field $h$.

The project combines full ED, matrix-free Lanczos, finite-size scaling,
$h$-field CQT/FOQT analysis, quench dynamics, spectral statistics and Majorana edge-mode visualization.

Both periodic boundary conditions (PBC) and open boundary conditions (OBC) are studied. PBC are used as the main reference for bulk critical estimates, while OBC provide boundary-condition diagnostics and access to edge-mode physics.

The code convention is

```math
H = -J \sum_{j=1}^{L_b} \sigma_j^z \sigma_{j+1}^z -g \sum_{j=1}^{L} \sigma_j^x -h \sum_{j=1}^{L} \sigma_j^z, \qquad J=1 .
```
For PBC, $L_b=L$ and $\sigma_{L+1}^z \equiv \sigma_1^z$.

For OBC, $L_b=L-1$.

## ✨ Key Features

- 🧮 **Full ED and Lanczos** for static observables, gaps and larger $h=0$ runs
- 📐 **Finite-size scaling** for critical-point and exponent estimates
- 🧲 **$h$-field CQT/FOQT** scaling analysis
- ⏱️ **Quench dynamics** for CQT, FOQT and Loschmidt protocols
- 🎲 **Spectral statistics**: $P(s)$, $P(r)$ and $\langle r\rangle(h)$
- 🧵 **Majorana edge-mode visualization** for OBC/topological intuition

## 🗂️ Directory Structure

```text
src/                         C sources
include/                     C headers
run/h_null/                  h=0 production shell drivers
scripts/h_null/              h=0 checks, observables and FSS plots
scripts/h_field/             h-field public plotting tools
scripts/h_field/compute/     h-field public grid driver
scripts/h_field/gh_surface/  g-h surface driver and plotters
scripts/quench/              quench driver and plotter
scripts/spectral/            minimal spectral plotter
data/                        generated data directory skeleton
plots/                       generated plots plus curated README assets
docs/                        reproducibility and pipeline notes
```

## 🚀 Getting Started

### C dependencies

```bash
# Ubuntu / Debian
sudo apt install gcc make liblapacke-dev libopenblas-dev

# Fedora / RHEL
sudo dnf install gcc make lapack-devel openblas-devel
```

### Python dependencies

```bash
pip install numpy scipy matplotlib seaborn
```

### Build

```bash
make clean && make all
```

## 🧰 Public Executables

| Executable | Purpose |
|---|---|
| `ising_static` | full ED $h=0$ / finite $h$ static observables |
| `ising_lanczos` | matrix-free Lanczos $h=0$ observables and gaps |
| `ising_chiz_fd` | finite-difference longitudinal susceptibility |
| `ising_hfield` | longitudinal $h$-field CQT/FOQT grids |
| `ising_gh_surface` | two-parameter $(g,h)$ surface worker |
| `ising_quench` | real-time ED quench dynamics |
| `ising_spectral` | ED-only spectral statistics raw data |

## 🔁 Workflow

<details>
<summary><b>h=0 ED, Lanczos and finite-size scaling</b></summary>

```bash
make static lanczos chizfd
bash run/h_null/run_pbc.sh
bash run/h_null/run_obc.sh
bash run/h_null/run_lanczos.sh --resume --campaign all
bash run/h_null/run_chifd_production.sh
python3 scripts/h_null/check_data_h_null.py
python3 scripts/h_null/plot_observables.py --lanczos
python3 scripts/h_null/fss_h_null_analysis.py --lanczos
```

</details>

<details>
<summary><b>h-field CQT/FOQT</b></summary>

```bash
make hfield
python3 scripts/h_field/compute/run_hfield_grid.py --resume
python3 scripts/h_field/plot_hfield.py
```

</details>

<details>
<summary><b>(g,h) surface</b></summary>

```bash
make ghsurface
python3 scripts/h_field/gh_surface/run_gh_surface.py --production-confirm --resume
python3 scripts/h_field/gh_surface/plot_gh_surface.py
python3 scripts/h_field/gh_surface/plot_gh_surface_collapse.py
```

</details>

<details>
<summary><b>Quench dynamics</b></summary>

```bash
make quench
bash scripts/quench/run_quench_ed.sh
python3 scripts/quench/plot_quench.py
```

</details>

<details>
<summary><b>Spectral statistics</b></summary>

```bash
make spectral
./ising_spectral --L 12 --g 1.0 --h 0.0 --omega 0.3 \
  --realization 0 --master-seed 12345 \
  --out data/spectral/raw/example.dat --overwrite
python3 scripts/spectral/plot_spectral.py
```

</details>

## 📊 Results

The table reports the final PBC estimates used as the main bulk critical estimates in the project report. OBC results are used throughout the analysis as boundary-condition diagnostics.

### Summary of critical exponents

| Quantity | PBC estimate | Exact value |
|---|---:|---:|
| $g_c$ | 0.999893 | 1 |
| $z$ | 1.0015 ± 0.0002 | 1 |
| $1/\nu$ | 0.9753 ± 0.0023 | 1 |
| $\beta/\nu$ | 0.1247 ± 0.0001 | $1/8 = 0.125$ |
| $\gamma/\nu$ | 1.7520 ± 0.0002 | $7/4 = 1.75$ |
| $v_F$ - sound velocity | 1.9968 | 2 |
| $\alpha$ | logarithmic, compatible with $0$ | $0$ |

Quoted uncertainties are residual $L_{\min}$ drifts, not statistical error bars.

## 🖼️ Figures

<details>
<summary><b>Ground-state observables</b></summary>
<br>

| Energy gaps $\Delta_0$ and $\Delta_1$ |
|:---:|
| <img src="plots/readme/h_null/delta_vs_g.png" width="700" alt="Energy gaps versus transverse field"><br>*Energy gaps Delta_0 and Delta_1 in function of transverse field g for PBC/OBC and several L.* |

| Low-energy spectrum at $L=22$ |
|:---:|
| <img src="plots/readme/h_null/gap_spectrum_panels.png" width="700" alt="Gap spectrum panels at L=22"><br>*Energy gaps Delta_0 and Delta_1 at L=22.* |

| Transverse magnetization $m_x$ |
|:---:|
| <img src="plots/readme/h_null/mx_vs_g.png" width="700" alt="Transverse magnetization versus transverse field"><br>*Ground-state transverse magnetization m_x(g) for PBC/OBC.* |

| Transverse susceptibility $g\chi_\perp$ |
|:---:|
| <img src="plots/readme/h_null/transverse_chi_panels.png" width="700" alt="Transverse susceptibility panels"><br>*Transverse susceptibility g chi_perp.* |

| Pseudo-order parameters $\tilde{\Psi}$ and $\bar{\Psi}$ |
|:---:|
| <img src="plots/readme/h_null/psi_panels.png" width="700" alt="Pseudo-order parameters"><br>*Pseudo-order parameters tilde Psi and bar Psi for PBC/OBC.* |

| Binder cumulant $U_4$ |
|:---:|
| <img src="plots/readme/h_null/binder_all.png" width="700" alt="Binder cumulant"><br>*Binder cumulant U_4(g), with crossings near g_c=1.* |

| Longitudinal susceptibility $\chi_z$ |
|:---:|
| <img src="plots/readme/h_null/chi_z.png" width="700" alt="Longitudinal susceptibility"><br>*Longitudinal susceptibility chi_z from finite differences around h=0.* |

</details>

<details>
<summary><b>FSS</b></summary>
<br>

| Scaled gap crossing |
|:---:|
| <img src="plots/readme/fss/gap_scaled_crossing.png" width="700" alt="Scaled gap crossing"><br>*Crossing of the scaled fundamental gap used to estimate g_pc.* |

| Sound velocity from finite-size gap |
|:---:|
| <img src="plots/readme/fss/kink_velocity_gap.png" width="700" alt="Sound velocity from gap"><br>*Sound velocity c extracted from critical gap scaling.* |

| Logarithmic transverse response |
|:---:|
| <img src="plots/readme/fss/alpha_chiperp_logfit.png" width="700" alt="Logarithmic transverse susceptibility fit"><br>*Logarithmic fits of C_x(L,g_pc)=g_pc chi_x(L,g_pc) for PBC and OBC.* |

| $1/\nu$ convergence | $\beta/\nu$ convergence |
|:---:|:---:|
| <img src="plots/readme/fss/nu_inv_vs_Lmin.png" width="400" alt="1/nu convergence"> | <img src="plots/readme/fss/beta_over_nu_vs_Lmin.png" width="400" alt="beta/nu convergence"> |
| **$z$ convergence** | **$\gamma/\nu$ convergence** |
| <img src="plots/readme/fss/z_vs_Lmin.png" width="400" alt="z convergence"> | <img src="plots/readme/fss/gamma_over_nu_vs_Lmin.png" width="400" alt="gamma/nu convergence"> |
| *Critical-exponent stability under L_min sweeps.* | *Critical-exponent stability under L_min sweeps.* |

| FSS data collapse |
|:---:|
| <img src="plots/readme/fss/fss_all.png" width="700" alt="FSS data collapse"><br>*FSS collapse using the final critical exponents and g_pc estimates.* |

</details>

<details>
<summary><b>h-field and FOQT/CQT</b></summary>
<br>

| CQT order parameter $m_z$ |
|:---:|
| <img src="plots/readme/hfield/cqt_order_parameter.png" width="700" alt="CQT order parameter"><br>*Longitudinal order parameter near the continuous quantum transition.* |

| CQT susceptibility $\chi_z$ |
|:---:|
| <img src="plots/readme/hfield/cqt_susceptibility.png" width="700" alt="CQT susceptibility"><br>*Longitudinal susceptibility and CQT scaling collapse.* |

| FOQT scaling |
|:---:|
| <img src="plots/readme/hfield/foqt_scaling.png" width="700" alt="FOQT scaling"><br>*First-order transition scaling in the ferromagnetic regime.* |

| Energy gaps $Delta_0(h)$ and $Delta_1(h)$ |
|:---:|
| <img src="plots/readme/hfield/gap_vs_h_ext.png" width="700" alt="Gaps versus longitudinal field"><br>*Energy gaps Delta_0(h) and Delta_1(h).* |

</details>
<details>
<summary><b>g-h surface analysis</b></summary>
<br>

| Gap surface $\Delta(g,h)$ |
|:---:|
| <img src="plots/readme/gh_surface/gap_surface_scaling_L14_pbc.png" width="700" alt="Gap surface scaling in the g-h plane"><br>*Fundamental gap surface on the two-parameter scaling grid.* |

| Longitudinal magnetization $m_z(g,h)$ |
|:---:|
| <img src="plots/readme/gh_surface/mz_surface_scaling_L14_pbc.png" width="700" alt="Longitudinal magnetization surface scaling in the g-h plane"><br>*Scaled longitudinal magnetization surface on the two-parameter grid.* |

</details>
<details>
<summary><b>Quench dynamics</b></summary>
<br>

| CQT dynamic FSS |
|:---:|
| <img src="plots/readme/quench/cqt_dynamic_fss.png" width="700" alt="CQT dynamic FSS"><br>*Soft-quench dynamics near the continuous quantum transition.* |

| FOQT dynamic scaling |
|:---:|
| <img src="plots/readme/quench/foqt_dynamic_scaling.png" width="700" alt="FOQT dynamic scaling"><br>*Longitudinal-field quench dynamics in the ferromagnetic regime.* |

| Loschmidt echo |
|:---:|
| <img src="plots/readme/quench/loschmidt_echo.png" width="700" alt="Loschmidt echo"><br>*Loschmidt echo after a longitudinal-field quench.* |

</details>

<details>
<summary><b>Spectral statistics</b></summary>
<br>

| Level spacing P(s), h=0 | Ratio distribution P(r), h=0 |
|:---:|:---:|
| <img src="plots/readme/spectral/level_spacing_distribution_h0.png" width="400" alt="Level spacing at h=0"><br>*Integrable h=0 spacing distribution.* | <img src="plots/readme/spectral/ratio_distribution_h0.png" width="400" alt="Ratio distribution at h=0"><br>*Integrable h=0 adjacent-gap ratio distribution.* |

| Level spacing P(s), h nonzero | Ratio distribution P(r), h nonzero |
|:---:|:---:|
| <img src="plots/readme/spectral/level_spacing_distribution_h_nonzero.png" width="400" alt="Level spacing at h nonzero"><br>*Level repulsion after integrability breaking.* | <img src="plots/readme/spectral/ratio_distribution_h_nonzero.png" width="400" alt="Ratio distribution at h nonzero"><br>*Adjacent-gap ratios after integrability breaking.* |

| Mean spacing ratio versus h |
|:---:|
| <img src="plots/readme/spectral/spacing_ratio_vs_h_omega0p3.png" width="700" alt="Spacing ratio versus h"><br>*Mean adjacent-gap ratio across the Poisson-to-GOE crossover.* |

</details>

<details>
<summary><b>Majorana edge modes</b></summary>
<br>

| Majorana edge-mode animation, $L=1000$ |
|:---:|
| <img src="plots/readme/majorana/majorana_github_readme.gif" width="700" alt="Majorana L=1000 animation"><br><em>Finite-chain Majorana edge-mode profile under OBC.</em> |

| Majorana profile, $L=1000$ | OBC gap decay $\Delta_0(L)$ |
|:---:|:---:|
| <img src="plots/readme/majorana/majorana_L1000.png" width="400" alt="Majorana L=1000 profile"><br><em>Spatial localization of the edge mode for a long open chain (L=22).</em> | <img src="plots/readme/majorana/delta0_obc_majorana.png" width="400" alt="OBC gap decay"><br><em>Exponential OBC gap decay in the ordered/topological regime (L=22).</em> |

</details>

## 📚 Bibliography

- P. Pfeuty, *The one-dimensional Ising model with a transverse field*, Annals of Physics (1970).
- S. Suzuki, J. Inoue and B. K. Chakrabarti, *Quantum Ising Phases and Transitions in Transverse Ising Models* (2013).
- M. Campostrini, A. Pelissetto and E. Vicari, *Finite-size scaling at quantum transitions*, Physical Review B (2014).
- D. Rossini and E. Vicari, *Coherent and dissipative dynamics at quantum phase transitions*, Physics Reports (2021).
- A. Y. Kitaev, *Unpaired Majorana fermions in quantum wires*, Physics-Uspekhi (2001).
- A. Maiellaro, F. Romeo and F. Illuminati, *Edge states, Majorana fermions, and topological order in superconducting wires with generalized boundary conditions*, Physical Review B (2022).

## 📄 License

This project is released under the [MIT License](LICENSE).
