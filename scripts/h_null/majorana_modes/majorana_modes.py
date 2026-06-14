#!/usr/bin/env python3
"""
Majorana figures for the OBC TFIM at zero longitudinal field.

Default outputs:
    plots/h_null/majorana/majorana_L22.pdf
    plots/h_null/majorana/majorana_L50.pdf
    plots/h_null/majorana/delta0_obc_majorana.pdf
    data/h_null/Majorana/majorana_gap_decay.dat

The BdG/free-fermion profiles are visual diagnostics of edge localization.
The many-body gap plot uses only OBC data from data/h_null/observables/OBC.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    mpl_cache_dir = Path("/tmp/qising_1d_matplotlib_cache/majorana")
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache_dir)

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
STATIC_DIR = PROJECT_ROOT / "data" / "h_null" / "observables" / "OBC"
PLOT_DIR = PROJECT_ROOT / "plots" / "h_null" / "majorana"
RESULTS_DIR = PROJECT_ROOT / "data" / "h_null" / "Majorana"
LEGACY_RESULTS_DIR = PROJECT_ROOT / "data" / "h_null" / "fss"

# Report defaults: one free-fermion profile and three ordered-phase gaps.
DEFAULT_PROFILE_G_VALUES = (0.5, 0.7, 0.85, 0.95)
DEFAULT_PROFILE_L = (22, 50)
DEFAULT_G_VALUES = (0.5, 0.7, 0.8)
DEFAULT_GAP_SIZES = (4, 6, 8, 10, 12, 14, 16, 18, 20, 22)
DEFAULT_LMIN_FIT = 10

AXIS_LABEL_FONTSIZE = 18
TICK_FONTSIZE = 15
LEGEND_FONTSIZE = 15
TEXT_FONTSIZE = 15

# Files from exploratory analyses are removed so the directory contains only
# the final Majorana section outputs plus optional media from majorana_media.py.
OBSOLETE_OUTPUTS = (
    "gap_obc_vs_L_residuals.pdf",
    "xi_fit_vs_exact.pdf",
    "xi_relative_deviation.pdf",
    "fit_quality_map.pdf",
    "gap_scaled_crossing.pdf",
    "gap_spectrum.pdf",
    "regime2_fit.pdf",
    "wavefunction_static.pdf",
    "majorana_profile_L22.pdf",
    "delta0_obc_semilog.pdf",
)


@dataclass(frozen=True)
class GapData:
    """One OBC gap table reduced to the fields needed for Delta_0(L,g)."""

    L: int
    backend: str
    path: Path
    g: np.ndarray
    delta0: np.ndarray


@dataclass(frozen=True)
class GapFitSummary:
    """Terminal summary for one representative g value."""

    g: float
    n_data: int
    n_fit: int
    fit_done: bool
    slope: float | None
    xi_m: float | None


def configure_style() -> None:
    """Use the same compact serif style adopted by the report figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "legend.fontsize": LEGEND_FONTSIZE,
    })


def save_fig(fig: plt.Figure, outname: str | Path) -> None:
    """Save a Matplotlib figure under plots/h_null/majorana unless given an absolute path."""
    path = Path(outname)
    if not path.is_absolute():
        path = PLOT_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path.relative_to(PROJECT_ROOT)}")


def cleanup_obsolete_outputs() -> None:
    """Remove outdated Majorana plots/tables that should not appear in the final set."""
    for name in OBSOLETE_OUTPUTS:
        path = PLOT_DIR / name
        if path.exists():
            path.unlink()
            print(f"  [cleanup] removed {path.relative_to(PROJECT_ROOT)}")
    old_table = RESULTS_DIR / "majorana_xi_fit.dat"
    if old_table.exists():
        old_table.unlink()
        print(f"  [cleanup] removed {old_table.relative_to(PROJECT_ROOT)}")
    for legacy_name in ("majorana_xi_fit.dat", "majorana_gap_decay.dat"):
        legacy_path = LEGACY_RESULTS_DIR / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()
            print(f"  [cleanup] removed {legacy_path.relative_to(PROJECT_ROOT)}")


def build_bdg_matrix(L: int, g: float) -> np.ndarray:
    """Return the 2L x 2L BdG matrix for the open TFIM/free-fermion chain."""
    if L < 2:
        raise ValueError("L must be >= 2")
    if g <= 0.0:
        raise ValueError("g must be positive")

    A = np.zeros((L, L), dtype=float)
    B = np.zeros((L, L), dtype=float)
    for i in range(L):
        A[i, i] = 2.0 * g
        if i < L - 1:
            # Open boundaries: nearest-neighbour hopping/pairing stops at L-1.
            A[i, i + 1] = -1.0
            A[i + 1, i] = -1.0
            B[i, i + 1] = -1.0
            B[i + 1, i] = 1.0
    return np.block([[A, B], [-B, -A]])


def majorana_profile_density(L: int, g: float) -> np.ndarray:
    """
    Site-resolved density of the quasi-zero BdG mode.

    For OBC and g<1 the two edge Majorana modes hybridise weakly at finite L.
    The closest-to-zero BdG eigenvector gives a symmetric edge-localized
    profile after combining particle and hole weights.

    The returned profile is rho_j = |u_j|^2 + |v_j|^2, normalized to one.
    The final symmetrization is only a visualization choice: it displays the
    equivalent left and right edge densities on equal footing in a clean chain.
    """
    evals, evecs = np.linalg.eigh(build_bdg_matrix(L, g))
    # The finite chain has a quasi-zero pair; the closest eigenvalue gives one
    # real-space representative of the edge-mode density.
    psi = evecs[:, int(np.argmin(np.abs(evals)))]
    rho = np.abs(psi[:L]) ** 2 + np.abs(psi[L:]) ** 2
    # The two edges are physically equivalent for a clean open chain.
    rho = 0.5 * (rho + rho[::-1])
    norm = float(np.sum(rho))
    if norm > 0.0:
        rho = rho / norm
    return rho


def plot_majorana_profile(
    L: int,
    g_values: tuple[float, ...] | list[float],
    outname: str | Path,
) -> None:
    """Plot several site-resolved BdG quasi-zero-mode profiles in one panel."""
    sites = np.arange(1, L + 1)
    colors = plt.cm.Blues(np.linspace(0.9, 0.25, len(g_values)))
    marker_step = max(1, L // 22)

    fig, ax = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
    for g, color in zip(g_values, colors):
        if g <= 0.0:
            print(f"  [WARN] skipping profile g={g}: BdG field must be positive")
            continue
        rho = majorana_profile_density(L, float(g))
        ax.fill_between(
            sites,
            0.0,
            rho,
            color=color,
            alpha=0.22,
            zorder=1,
        )
        ax.plot(
            sites,
            rho,
            "-o",
            color=color,
            lw=1.5,
            ms=3.0,
            markevery=marker_step,
            label=rf"$g={g:.2f}$",
            zorder=2,
        )

    ax.set_xlabel(r"Site $j$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(r"$\rho_j$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xlim(0.5, L + 0.5)
    ax.set_ylim(bottom=0.0)
    if L == 22:
        ax.set_xticks(list(range(2, L + 1, 2)))
    ax.tick_params(direction="in", which="both", top=True, right=True, labelsize=TICK_FONTSIZE)
    ax.grid(alpha=0.28, linestyle=":")
    ax.legend(frameon=False, loc="upper center", fontsize=LEGEND_FONTSIZE)
    save_fig(fig, outname)


def _gap_file_for_l(static_dir: Path, L: int) -> tuple[str, Path, int]:
    """Map a system size to its OBC gap file and Delta_0 column."""
    if L <= 12:
        return "ED", static_dir / f"gap_obc_L{L:02d}.dat", 5
    return "LNCZ", static_dir / f"gap_lz_obc_L{L:02d}.dat", 4


def load_obc_gap_data(
    static_dir: Path = STATIC_DIR,
    sizes: tuple[int, ...] | list[int] = DEFAULT_GAP_SIZES,
) -> dict[int, GapData]:
    """Load OBC many-body gaps, using ED for L<=12 and Lanczos for L>12."""
    gap_data: dict[int, GapData] = {}
    for L in sizes:
        backend, path, gap_col = _gap_file_for_l(static_dir, int(L))
        if not path.exists():
            print(f"  [WARN] missing OBC gap file for L={L:02d}: {path.relative_to(PROJECT_ROOT)}")
            continue
        try:
            arr = np.loadtxt(path, comments="#")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"  [WARN] cannot read {path.relative_to(PROJECT_ROOT)}: {exc}")
            continue
        arr = np.atleast_2d(arr)
        if arr.ndim != 2 or arr.shape[1] <= gap_col:
            print(f"  [WARN] invalid schema in {path.relative_to(PROJECT_ROOT)}")
            continue
        # Keep only rows that can contribute to interpolation in g.
        finite = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, gap_col])
        arr = arr[finite]
        if arr.size == 0:
            print(f"  [WARN] no finite OBC gap rows in {path.relative_to(PROJECT_ROOT)}")
            continue
        order = np.argsort(arr[:, 0])
        gap_data[int(L)] = GapData(
            L=int(L),
            backend=backend,
            path=path,
            g=arr[order, 0],
            delta0=arr[order, gap_col],
        )
    return dict(sorted(gap_data.items()))


def interpolate_delta0(data: GapData, g_value: float) -> float | None:
    """Linearly interpolate Delta_0 at g, rejecting out-of-range or invalid values."""
    if g_value < float(np.min(data.g)) or g_value > float(np.max(data.g)):
        return None
    delta = float(np.interp(g_value, data.g, data.delta0))
    if not np.isfinite(delta) or delta <= 0.0:
        return None
    return delta


def fit_log_delta(
    L_values: np.ndarray,
    delta_values: np.ndarray,
    Lmin_fit: int,
) -> tuple[float, float, float] | None:
    """Fit log(Delta_0)=intercept+slope*L on the large-L tail."""
    mask = (L_values >= Lmin_fit) & np.isfinite(delta_values) & (delta_values > 0.0)
    if np.count_nonzero(mask) < 2:
        return None
    slope, intercept = np.polyfit(L_values[mask], np.log(delta_values[mask]), deg=1)
    if not np.isfinite(slope) or slope >= 0.0:
        return None
    xi_m = -1.0 / float(slope)
    return float(slope), float(intercept), xi_m


def plot_delta0_obc_semilog(
    gap_data: dict[int, GapData],
    g_values: tuple[float, ...] | list[float] = DEFAULT_G_VALUES,
    outname: str | Path = "delta0_obc_majorana.pdf",
    Lmin_fit: int = DEFAULT_LMIN_FIT,
) -> list[GapFitSummary]:
    """
    Plot Delta_0(L,g)=E_1-E_0 for OBC and g<1.

    For OBC and g<1, the lowest many-body gap is interpreted as the
    finite-size splitting produced by the overlap of the two edge Majorana
    modes. The exponential fit is used only as a diagnostic of edge-mode
    hybridisation; no statistical uncertainty is assigned to deterministic
    ED/Lanczos data.
    """
    colors = plt.cm.Blues(np.linspace(0.9, 0.25, len(g_values)))
    fig, ax = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
    summaries: list[GapFitSummary] = []
    fit_rows: list[tuple[float, int, int, float, float]] = []

    for g_value, color in zip(g_values, colors):
        if not (0.0 < g_value < 1.0):
            print(f"  [WARN] skipping g={g_value}: Majorana semilog plot uses only g<1")
            continue

        L_list: list[int] = []
        delta_list: list[float] = []
        for L, data in gap_data.items():
            delta = interpolate_delta0(data, g_value)
            if delta is None:
                continue
            L_list.append(L)
            delta_list.append(delta)

        L_arr = np.asarray(L_list, dtype=float)
        delta_arr = np.asarray(delta_list, dtype=float)
        if L_arr.size == 0:
            print(f"  [WARN] g={g_value:.3f}: no OBC gap file covers this g value; no points plotted")
            summaries.append(GapFitSummary(g_value, 0, 0, False, None, None))
            continue

        # Markers use all valid available sizes. Dashed curves use only the
        # large-L tail where edge-mode overlap is expected to dominate.
        low = L_arr < float(Lmin_fit)
        high = ~low
        if np.count_nonzero(low):
            ax.semilogy(
                L_arr[low],
                delta_arr[low],
                "o",
                color=color,
                ms=5.0,
                markerfacecolor="none",
                markeredgewidth=1.1,
                label=rf"$g={g_value:.1f}$",
            )
        if np.count_nonzero(high):
            ax.semilogy(
                L_arr[high],
                delta_arr[high],
                "o",
                color=color,
                ms=5.0,
                label="_nolegend_" if np.count_nonzero(low) else rf"$g={g_value:.1f}$",
            )

        fit = fit_log_delta(L_arr, delta_arr, Lmin_fit)
        if fit is None:
            n_fit = int(np.count_nonzero(L_arr >= Lmin_fit))
            summaries.append(GapFitSummary(g_value, int(L_arr.size), n_fit, False, None, None))
            continue

        slope, intercept, xi_m = fit
        n_fit = int(np.count_nonzero(L_arr >= Lmin_fit))
        L_fit = np.linspace(float(np.min(L_arr[L_arr >= Lmin_fit])), float(np.max(L_arr)), 160)
        ax.semilogy(
            L_fit,
            np.exp(intercept + slope * L_fit),
            "--",
            color=color,
            lw=1.5,
            alpha=0.95,
            label="_nolegend_",
        )
        fit_rows.append((g_value, Lmin_fit, n_fit, slope, xi_m))
        summaries.append(GapFitSummary(g_value, int(L_arr.size), n_fit, True, slope, xi_m))

    ax.set_xlabel(r"$L$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(r"$\Delta_0(L,g)$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.grid(alpha=0.28, linestyle=":", which="both")
    ax.tick_params(direction="in", which="both", top=True, right=True, labelsize=TICK_FONTSIZE)
    ax.legend(frameon=False, loc="lower left", fontsize=LEGEND_FONTSIZE)
    save_fig(fig, outname)
    write_gap_decay_table(fit_rows)
    return summaries


def write_gap_decay_table(rows: list[tuple[float, int, int, float, float]]) -> None:
    """Write the minimal deterministic fit table used by the semilog figure."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "majorana_gap_decay.dat"
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Majorana gap-decay diagnostic for OBC TFIM, h=0\n")
        handle.write("# No statistical uncertainties are assigned to deterministic ED/Lanczos data.\n")
        handle.write("# g  Lmin_fit  npts  slope_log_delta  xi_M_fit\n")
        for g_value, lmin, npts, slope, xi_m in rows:
            handle.write(f"{g_value:.8f}  {lmin:d}  {npts:d}  {slope:.12e}  {xi_m:.12e}\n")
    print(f"  [OK] {path.relative_to(PROJECT_ROOT)}")


def parse_float_list(raw: str | list[str] | tuple[float, ...]) -> tuple[float, ...]:
    """Parse comma- or whitespace-separated floats from CLI input."""
    if isinstance(raw, tuple):
        return tuple(float(value) for value in raw)
    raw_items = raw if isinstance(raw, list) else [raw]
    parts: list[str] = []
    for item in raw_items:
        parts.extend(part.strip() for part in re.split(r"[,\s]+", str(item)) if part.strip())
    return tuple(float(part) for part in parts)


def parse_int_list(raw: str | list[str] | tuple[int, ...]) -> tuple[int, ...]:
    """Parse comma- or whitespace-separated integers from CLI input."""
    if isinstance(raw, tuple):
        return tuple(int(value) for value in raw)
    raw_items = raw if isinstance(raw, list) else [raw]
    parts: list[str] = []
    for item in raw_items:
        parts.extend(part.strip() for part in re.split(r"[,\s]+", str(item)) if part.strip())
    return tuple(int(part) for part in parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the three final OBC Majorana figures for the TFIM/ED project.",
    )
    parser.add_argument("--profile-g-values", nargs="+", default=DEFAULT_PROFILE_G_VALUES,
                        help="Profile g values, comma- or whitespace-separated (default: 0.5 0.7 0.85 0.95).")
    parser.add_argument("--profile-L", nargs="+", default=DEFAULT_PROFILE_L,
                        help="Profile sizes, comma- or whitespace-separated (default: 22).")
    parser.add_argument("--g-values", nargs="+", default=DEFAULT_G_VALUES,
                        help="g values for Delta0 semilog plot (default: 0.5 0.7 0.8).")
    parser.add_argument("--gap-sizes", nargs="+", default=DEFAULT_GAP_SIZES,
                        help="OBC many-body sizes to try loading (default: 4,6,...,22).")
    parser.add_argument("--Lmin-fit", type=int, default=DEFAULT_LMIN_FIT)
    parser.add_argument("--static-dir", type=Path, default=STATIC_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile_g_values = parse_float_list(args.profile_g_values)
    profile_L = parse_int_list(args.profile_L)
    gap_g_values = parse_float_list(args.g_values)
    gap_sizes = parse_int_list(args.gap_sizes)

    configure_style()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[majorana_modes.py]  Cleaning obsolete Majorana outputs ...")
    cleanup_obsolete_outputs()

    print("[majorana_modes.py]  Generating BdG Majorana profiles ...")
    for L in profile_L:
        plot_majorana_profile(
            L=int(L),
            g_values=profile_g_values,
            outname=f"majorana_L{int(L):02d}.pdf",
        )

    print("[majorana_modes.py]  Loading OBC gap data ...")
    gap_data = load_obc_gap_data(static_dir=args.static_dir, sizes=gap_sizes)
    if not gap_data:
        raise SystemExit("[ERROR] No OBC gap data found in data/h_null/observables/OBC.")
    used_L = sorted(gap_data)
    print("  Available OBC L used:", ", ".join(str(L) for L in used_L))

    print("[majorana_modes.py]  Generating Delta0 semilog plot ...")
    summaries = plot_delta0_obc_semilog(
        gap_data=gap_data,
        g_values=gap_g_values,
        Lmin_fit=args.Lmin_fit,
    )

    print("\nSUMMARY")
    print("-------")
    print("Majorana profile g values:", ", ".join(f"{g:.6g}" for g in profile_g_values))
    print("Delta0 OBC L used:", ", ".join(str(L) for L in used_L))
    for item in summaries:
        if item.fit_done:
            print(
                f"g={item.g:.3f}: exponential fit done with {item.n_fit} points "
                f"(xi_M fit={item.xi_m:.4g})"
            )
        else:
            if item.n_data == 0:
                print(f"g={item.g:.3f}: no data in the available OBC g ranges")
            else:
                print(f"g={item.g:.3f}: data plotted, no large-L tail fit ({item.n_fit} points with L>= {args.Lmin_fit})")
    print("\n[majorana_modes.py]  Done. Profile/gap PDF files and 1 diagnostic table written.")


if __name__ == "__main__":
    main()
