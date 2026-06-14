#!/usr/bin/env python3
"""
Finite-size scaling analysis for deterministic ED/Lanczos data of the
one-dimensional quantum Ising chain.

Inputs are read from data/h_null/observables and data/h_null/chiz_fd.
Derived tables are written to data/h_null/fss, while publication figures are
written to plots/h_null/fss.

Usage: python3 scripts/h_null/fss_h_null_analysis.py [--lanczos] [--L LIST]
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    mpl_cache_dir = Path("/tmp/qising_1d_matplotlib_cache/h_null")
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache_dir)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import brentq, curve_fit, OptimizeWarning

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OBSERVABLES_DIR = PROJECT_ROOT / "data" / "h_null" / "observables"
CHIZ_FD_DIR = PROJECT_ROOT / "data" / "h_null" / "chiz_fd"
FSS_DIR = PROJECT_ROOT / "data" / "h_null" / "fss"
FSS_PBC_DIR = FSS_DIR / "PBC"
FSS_OBC_DIR = FSS_DIR / "OBC"
PLOT_DIR = PROJECT_ROOT / "plots" / "h_null" / "fss"
PSI_FSS_DIR = PLOT_DIR / "Psi"
FSS_DIR.mkdir(parents=True, exist_ok=True)
FSS_PBC_DIR.mkdir(parents=True, exist_ok=True)
FSS_OBC_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
PSI_FSS_DIR.mkdir(parents=True, exist_ok=True)

G_C = 1.0
C_EXACT = 2.0

# Asymptotes of Delta0(g_c,L)*L for the lattice TFIM.
GAP_L_CFT_PBC = np.pi / 2.0
GAP_L_CFT_OBC = np.pi

ED_SIZES = [4, 6, 8, 10, 12]
LZ_SIZES = [14, 16, 18, 20, 22]
ALL_SIZES = ED_SIZES + LZ_SIZES
PBC_EXTRA_SIZES = []
PBC_SIZES = ALL_SIZES + PBC_EXTRA_SIZES
OBC_SIZES = ALL_SIZES
LMIN_SWEEP = [4, 6, 8, 10, 12]
FINAL_LMIN_DEFAULT = 10
OBC_LE_SHIFT = 0.5
GAP_RESIDUAL_OMEGA = 2.0
OBC_RAW_OMEGA = 1.0

FONT_SCALE = 1.6
AXIS_LABEL_FONTSIZE = 18
TICK_FONTSIZE = 15
LEGEND_FONTSIZE = 15
TITLE_FONTSIZE = 15
TEXT_FONTSIZE = 15
INSET_TICK_FONTSIZE = 12
EXPONENT_CONVERGENCE_LEGEND_FONTSIZE = 9
EXPONENT_CONVERGENCE_LEGEND_Y_ANCHOR = -0.08
EXPONENT_CONVERGENCE_LEGEND_LABEL_XPAD = 0.012
EXPONENT_CONVERGENCE_LEGEND_X_CENTER = 0.57
EXPONENT_CONVERGENCE_LEGEND_X_SPACING = 0.30
EXTERNAL_LEGEND_Y_ANCHOR = -0.15
GC_EXACT: float = 1.0
# Set this according to how mz_sq is written in the observable files.
_MZ_SQ_NORMALISED: bool = True

_cmap_pbc = plt.cm.plasma
_cmap_obc = plt.cm.viridis
_color_values = np.linspace(0.0, 0.85, len(ALL_SIZES))
COLORS_PBC = {L: _cmap_pbc(v) for L, v in zip(ALL_SIZES, _color_values)}
COLORS_OBC = {L: _cmap_obc(v) for L, v in zip(ALL_SIZES, _color_values)}
_extra_color_values = np.linspace(0.88, 0.98, len(PBC_EXTRA_SIZES))
for L, v in zip(PBC_EXTRA_SIZES, _extra_color_values):
    if L not in COLORS_PBC:
        COLORS_PBC[L] = _cmap_pbc(v)

def colors_for_bc(bc_label: str) -> dict[int, tuple]:
    return COLORS_PBC if bc_label.upper() == "PBC" else COLORS_OBC


class _DoubleBCLine:
    def __init__(self, L: int, linestyle: str = "-") -> None:
        self.L = int(L)
        self.linestyle = linestyle


class _DoubleBCLineHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        y_pbc = y0 + 0.68 * height
        y_obc = y0 + 0.32 * height
        artists = [
            Line2D(
                [x0, x0 + width], [y_pbc, y_pbc],
                color=COLORS_PBC.get(orig_handle.L, "gray"),
                ls=orig_handle.linestyle,
                lw=1.9,
                solid_capstyle="round",
            ),
            Line2D(
                [x0, x0 + width], [y_obc, y_obc],
                color=COLORS_OBC.get(orig_handle.L, "gray"),
                ls=orig_handle.linestyle,
                lw=1.9,
                solid_capstyle="round",
            ),
        ]
        for artist in artists:
            artist.set_transform(handlebox.get_transform())
            handlebox.add_artist(artist)
        return artists[0]


def _method_code_from_source(source, L: int) -> int:
    if isinstance(source, str):
        source_upper = source.upper()
        if source_upper == "ED":
            return 0
        if source_upper in {"LNCZ", "LANCZOS", "LZ"}:
            return 1
    try:
        method = int(source)
        if method in (0, 1):
            return method
    except (TypeError, ValueError):
        pass
    return 0 if int(L) <= ED_SIZES[-1] else 1


def _register_external_legend_item(
    plotted_L_methods: dict[int, set[int]],
    plotted_methods: set[int],
    L: int,
    method,
) -> None:
    method_code = _method_code_from_source(method, int(L))
    plotted_L_methods.setdefault(int(L), set()).add(method_code)
    if method_code in (0, 1):
        plotted_methods.add(method_code)


def _linestyle_for_method_set(methods: set[int]) -> str:
    if 1 in methods:
        return "--"
    return "-"


def _external_size_method_legend(
    fig: plt.Figure,
    plotted_L_methods: dict[int, set[int]],
    plotted_methods: set[int],
    *,
    encode_method_in_size: bool = True,
    y_anchor: float = EXTERNAL_LEGEND_Y_ANCHOR,
) -> None:
    handles = []
    labels = []
    for L in ALL_SIZES:
        if L not in plotted_L_methods:
            continue
        linestyle = _linestyle_for_method_set(plotted_L_methods[L]) if encode_method_in_size else "-"
        handles.append(_DoubleBCLine(L, linestyle))
        labels.append(rf"$L={L}$")

    if 0 in plotted_methods:
        handles.append(Line2D([], [], color="gray", ls="-", lw=1.9))
        labels.append("ED")
    if 1 in plotted_methods:
        handles.append(Line2D([], [], color="gray", ls="--", lw=1.9))
        labels.append("LNCZ")

    if not handles:
        return

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, y_anchor),
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        ncol=6,
        columnspacing=1.7,
        handlelength=2.2,
        handletextpad=0.8,
        handler_map={_DoubleBCLine: _DoubleBCLineHandler()},
    )


def fss_dir_for_bc(bc_label: str) -> Path:
    return FSS_PBC_DIR if str(bc_label).upper() == "PBC" else FSS_OBC_DIR


def fss_path_for_bc(bc_label: str, name: str) -> Path:
    path = fss_dir_for_bc(bc_label) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

COL = dict(g=0, Mx=1, mz_sq=2, mz=3, chi_z=4,
           mz4=5, psi_tilde=6, psi_bar=7,
           binder=8, chi_perp=9, g_chi_perp=10)
CHIZFD_COL = dict(g=0, dh=1, method_code=2, chi_fd=7, oddness1=8, oddness2=9)

# Exact critical exponents for the 1D quantum Ising universality class.
NU = 1.0        # correlation length exponent
OMEGA_PBC = 2.0
OMEGA_OBC = 1.0
Z_EXACT = 1.0  # dynamic exponent (quantum Ising, z=1)
BETA = 0.125   # magnetization exponent (= 1/8)
GAMMA = 1.75   # susceptibility exponent (= 7/4)

Z_DYN = Z_EXACT

def _fit_gap_L_fixed_omega(
    L_arr: np.ndarray,
    gapL_arr: np.ndarray,
    omega: float,
) -> dict:
    """
    Two-parameter linear fit with fixed omega:
        gapL(L) = C * (1 + B * L^{-omega})
    =>  gapL = C + (C*B) * L^{-omega}
    =>  y = a + b * x  with x = L^{-omega}

    Returns:
        C, B, omega (fixed correction exponent)
    """
    L_arr = np.asarray(L_arr, dtype=float)
    gapL_arr = np.asarray(gapL_arr, dtype=float)

    x = L_arr ** (-omega)
    A_mat = np.column_stack([np.ones_like(x), x])
    result = np.linalg.lstsq(A_mat, gapL_arr, rcond=None)
    coeffs = result[0]            # [a, b]
    C_hat = coeffs[0]
    B_hat = coeffs[1] / C_hat if abs(C_hat) > 1e-15 else np.nan  # B = b/C

    return dict(C=float(C_hat), B=float(B_hat), omega=float(omega))


# ---------------------------------------------------------------------------
# Shared data helpers
# ---------------------------------------------------------------------------
def gap_columns(gd: np.ndarray) -> dict:
    gd = np.atleast_2d(gd)
    nc = gd.shape[1]
    gap_col = 5 if nc == 8 else 4
    gapL_col = 6 if nc == 8 else 5
    g_arr = gd[:, 0]
    E0 = gd[:, 1]
    Delta0 = gd[:, gap_col]
    Delta1 = gd[:, 3] - gd[:, 1]
    return {
        "nc": nc,
        "gap_col": gap_col,
        "gapL_col": gapL_col,
        "g_arr": g_arr,
        "E0": E0,
        "Delta0": Delta0,
        "Delta1": Delta1,
    }


def data_dir_for_bc(pbc: bool) -> Path:
    return OBSERVABLES_DIR / ("PBC" if pbc else "OBC")


def load_obs_data(sizes, bc):
    """
    Returns {L: (g_arr, full_obs_matrix)}.
    Tries obs_lz_<bc>_L*.dat first (Lanczos); falls back to obs_<bc>_L*.dat (ED).
    Skips files with fewer than 11 columns (pre-psi_bar schema).
    """
    data = {}
    datadir = data_dir_for_bc(bc != "obc")
    for L in sizes:
        if bc == "obc":
            candidates = [
                Path(datadir) / f"obs_lz_obc_L{L:02d}.dat",
                Path(datadir) / f"obs_obc_L{L:02d}.dat",
            ]
        else:
            candidates = [
                Path(datadir) / f"obs_lz_L{L:02d}.dat",
                Path(datadir) / f"obs_L{L:02d}.dat",
            ]
        for fname in candidates:
            if fname.exists():
                try:
                    od = np.atleast_2d(np.loadtxt(fname, comments="#"))
                    od = od[np.isfinite(od).all(axis=1)]
                    if od.shape[1] < 11:
                        print(f"  SKIP {fname.name}: only {od.shape[1]} cols "
                              f"(need 11, missing psi_bar?)")
                        continue
                    data[L] = (od[:, COL["g"]], od)
                    break
                except Exception as e:
                    print(f"  WARN {fname.name}: {e}")
    return data


def _read_chiz_fd_header(path: Path) -> dict[str, str]:
    meta = {}
    try:
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                if not line.startswith("#"):
                    continue
                body = line[1:].strip()
                if "=" not in body:
                    continue
                key, value = body.split("=", 1)
                meta[key.strip()] = value.strip()
    except OSError as exc:
        print(f"  WARN {path.name}: cannot read header: {exc}")
    return meta


def load_chiz_fd_data(
    dh_tag: str = "dh_5e-04",
    sizes=ALL_SIZES,
    bc: str = "pbc",
    allow_partial: bool = False,
):
    """Load chi_z finite-difference files generated by ising_chiz_fd."""
    data = {}
    bc_key = str(bc).lower()
    bc_label = "OBC" if bc_key == "obc" else "PBC"
    base_dir = CHIZ_FD_DIR / dh_tag / bc_label

    for L in sizes:
        fname = f"chizfd_obc_L{L:02d}.dat" if bc_key == "obc" else f"chizfd_L{L:02d}.dat"
        path = base_dir / fname
        if not path.exists():
            print(f"  SKIP chiz_fd {bc_label} L={L:02d}: file not found ({path})")
            continue

        meta = _read_chiz_fd_header(path)
        try:
            n_expected = int(meta["n_g"]) if "n_g" in meta else None
        except ValueError:
            print(f"  WARN chiz_fd {bc_label} L={L:02d}: invalid header n_g={meta.get('n_g')!r}")
            n_expected = None

        try:
            arr = np.genfromtxt(path, comments="#")
        except Exception as exc:
            print(f"  WARN chiz_fd {bc_label} L={L:02d}: cannot load {path.name}: {exc}")
            continue
        arr = np.atleast_2d(arr)
        if arr.size == 0 or arr.shape[1] == 0 or not np.isfinite(arr).any():
            print(f"  WARN chiz_fd {bc_label} L={L:02d}: no numeric rows in {path.name}")
            continue
        if arr.shape[1] != 10:
            print(f"  WARN chiz_fd {bc_label} L={L:02d}: NF={arr.shape[1]} (expected 10), skipped")
            continue

        finite_rows = np.isfinite(arr).all(axis=1)
        if not np.all(finite_rows):
            print(f"  WARN chiz_fd {bc_label} L={L:02d}: dropping {np.count_nonzero(~finite_rows)} non-finite rows")
            arr = arr[finite_rows]
        if arr.shape[0] == 0:
            print(f"  WARN chiz_fd {bc_label} L={L:02d}: no finite rows after filtering")
            continue

        n_rows = int(arr.shape[0])
        n_expected_eff = int(n_expected) if n_expected is not None else n_rows
        complete = bool(n_rows == n_expected_eff)
        if not complete and not allow_partial:
            print(
                f"  WARN chiz_fd {bc_label} L={L:02d}: partial {n_rows}/{n_expected_eff}, "
                "skipped (use --allow-partial-chizfd)"
            )
            continue

        method_vals = np.unique(arr[:, CHIZFD_COL["method_code"]].astype(int))
        method_code = int(method_vals[0]) if method_vals.size else -1
        if method_vals.size > 1:
            print(f"  WARN chiz_fd {bc_label} L={L:02d}: multiple method_code values {method_vals}")
        expected_method = 0 if int(L) <= ED_SIZES[-1] else 1
        if method_code != expected_method:
            print(
                f"  WARN chiz_fd {bc_label} L={L:02d}: method_code={method_code}, "
                f"expected {expected_method}"
            )

        chi = arr[:, CHIZFD_COL["chi_fd"]]
        g_arr = arr[:, CHIZFD_COL["g"]]
        positive = chi > 0.0
        if not np.all(positive):
            print(f"  WARN chiz_fd {bc_label} L={L:02d}: {np.count_nonzero(~positive)} non-positive chi_fd rows")
        plot_mask = np.isfinite(g_arr) & np.isfinite(chi) & positive
        if np.count_nonzero(plot_mask) < 1:
            print(f"  WARN chiz_fd {bc_label} L={L:02d}: no positive finite chi_fd rows")
            continue

        chi_plot = chi[plot_mask]
        g_plot = g_arr[plot_mask]
        max_odd1 = float(np.nanmax(arr[:, CHIZFD_COL["oddness1"]]))
        max_odd2 = float(np.nanmax(arr[:, CHIZFD_COL["oddness2"]]))
        status = "complete" if complete else "partial"
        print(
            f"  chiz_fd {bc_label} L={L:02d}: rows={n_rows} expected={n_expected_eff} "
            f"{status} chi_min={np.nanmin(chi):.12e} chi_max={np.nanmax(chi):.12e} "
            f"max_oddness1={max_odd1:.12e} max_oddness2={max_odd2:.12e}"
        )

        data[int(L)] = {
            "g": g_plot,
            "chi": chi_plot,
            "method": method_code,
            "complete": complete,
            "n_expected": n_expected_eff,
            "n_rows": n_rows,
            "path": path,
        }

    return data


def save_fig(fig: plt.Figure, name: str) -> None:
    path = PLOT_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path.name}")


def save_psi_fig(fig: plt.Figure, name: str) -> None:
    path = PSI_FSS_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Psi/{path.name}")


def apply_grid(ax: plt.Axes) -> None:
    """Apply the shared internal plot grid style."""
    ax.grid(True, which="major", axis="both", color="0.70",
            alpha=0.55, linestyle=":", linewidth=0.8, zorder=0)


def apply_plain_decimal_y_ticks(ax: plt.Axes, ticks=None) -> None:
    """Format y tick labels as plain numbers with at most one decimal digit."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    def _fmt(value, _pos):
        if not np.isfinite(value) or value <= 0.0:
            return ""
        return f"{value:.1f}".rstrip("0").rstrip(".")

    formatter = FuncFormatter(_fmt)
    if ticks is not None:
        ax.yaxis.set_major_locator(FixedLocator([float(t) for t in ticks if t > 0.0]))
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.offsetText.set_visible(False)


def apply_two_decimal_y_ticks(ax: plt.Axes) -> None:
    """Format y tick labels with two decimal digits."""
    from matplotlib.ticker import FuncFormatter, NullFormatter

    def _fmt(value, _pos):
        if not np.isfinite(value):
            return ""
        return f"{value:.2f}"

    ax.yaxis.set_major_formatter(FuncFormatter(_fmt))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.offsetText.set_visible(False)


def plot_chiz_fd_fss_panels(
    chiz_pbc,
    chiz_obc,
    g_c_num: float = 1.0,
    g_c_num_obc: float | None = None,
    nu_num: float = 1.0,
    nu_num_obc: float | None = None,
    gamma_over_nu: float = 1.75,
    gamma_over_nu_obc: float | None = None,
) -> None:
    """FSS/collapse panels for official finite-difference chi_z data only."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullLocator, NullFormatter, FuncFormatter
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True,
                             sharey=True)
    panels = [
        (axes[0], "PBC", chiz_pbc),
        (axes[1], "OBC", chiz_obc),
    ]
    plotted_L_methods: dict[int, set[int]] = {}
    plotted_methods: set[int] = set()
    if g_c_num_obc is None:
        g_c_num_obc = g_c_num
    if nu_num_obc is None:
        nu_num_obc = nu_num
    if gamma_over_nu_obc is None:
        gamma_over_nu_obc = gamma_over_nu

    for ax, bc_label, dataset in panels:
        g_ref = g_c_num if bc_label == "PBC" else g_c_num_obc
        nu_ref = nu_num if bc_label == "PBC" else nu_num_obc
        gamma_ref = gamma_over_nu if bc_label == "PBC" else gamma_over_nu_obc
        colors = colors_for_bc(bc_label)
        has_data = False
        has_ed = False
        has_lz = False
        axins = inset_axes(ax,
                           width="40%",
                           height="40%",
                           loc="upper left",
                           bbox_to_anchor=(0.47, 0.01, 1.0, 1.0),
                           bbox_transform=ax.transAxes,
                           borderpad=1.1)
        zoom_y_vals = []
        zoom_has_data = False

        for L in sorted(dataset.keys()):
            if int(L) < FINAL_LMIN_DEFAULT:
                continue
            item = dataset[L]
            g_arr = np.asarray(item["g"], dtype=float)
            chi = np.asarray(item["chi"], dtype=float)
            m = np.isfinite(g_arr) & np.isfinite(chi) & (chi > 0.0)
            if np.count_nonzero(m) < 1:
                continue

            method = int(item.get("method", 0 if int(L) <= ED_SIZES[-1] else 1))
            complete = bool(item.get("complete", True))
            ls = "-" if method == 0 else "--"
            has_ed = has_ed or method == 0
            has_lz = has_lz or method == 1
            label = rf"$L={L}$"
            if not complete:
                ls = ":"
                label = rf"$L={L}$ partial {item.get('n_rows', len(g_arr))}/{item.get('n_expected', '?')}"

            x = (g_arr[m] - g_ref) * (float(L) ** (1.0 / nu_ref))
            y = chi[m] * (float(L) ** (-gamma_ref))
            keep = np.isfinite(x) & np.isfinite(y) & (y > 0.0)
            keep &= (x >= -4.0) & (x <= 4.0)
            x = x[keep]
            y = y[keep]
            if x.size < 1:
                continue

            ax.plot(x, y, color=colors.get(L, "gray"), ls=ls, lw=1.35, label=label)
            _register_external_legend_item(plotted_L_methods, plotted_methods, int(L), method)
            mz = (x >= -0.2) & (x <= 0.2)
            if np.count_nonzero(mz) >= 2:
                axins.plot(x[mz], y[mz], color=colors.get(L, "gray"), ls=ls, lw=1.05)
                zoom_y_vals.extend(y[mz].tolist())
                zoom_has_data = True
            has_data = True

        ax.axvline(0.0, color="gray", ls=":", lw=1.2, alpha=0.7)
        ax.set_xlim(-4, 4)
        ax.set_yscale("log")
        ax.set_xlabel(r"$(g-g_{pc})L^{1/\nu}$")
        if ax is axes[0]:
            ax.set_ylabel(r"$\chi_z L^{-\gamma/\nu}$")
        ax.set_title(bc_label, loc="right", fontsize=TITLE_FONTSIZE)

        ax.tick_params(direction="in", which="both", top=True, right=True, labelsize=TICK_FONTSIZE)
        apply_grid(ax)
        if not has_data:
            ax.text(0.5, 0.5, "No finite FD data",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=TEXT_FONTSIZE, color="gray")
            axins.set_visible(False)
            continue
        if zoom_has_data:
            axins.axvline(0.0, color="gray", ls=":", lw=1.0, alpha=0.65)
            axins.set_xlim(-0.06, 0.06)
            axins.set_yscale("log")
            zoom_y = np.asarray(zoom_y_vals, dtype=float)
            zoom_y = zoom_y[np.isfinite(zoom_y) & (zoom_y > 0.0)]
            if zoom_y.size:
                ylo = float(np.min(zoom_y))
                yhi = float(np.max(zoom_y))
                if yhi > ylo:
                    axins.set_ylim(ylo / 1.08, yhi * 1.08)
            apply_two_decimal_y_ticks(axins)
            axins.tick_params(direction="in", which="both", top=True, right=True,
                              labelsize=TICK_FONTSIZE)
            apply_grid(axins)
        else:
            axins.set_visible(False)
    _external_size_method_legend(fig, plotted_L_methods, plotted_methods)
    save_fig(fig, "chiz_fss.pdf")


def plot_chiperp_fss_panels(
    obs_pbc,
    obs_obc,
    g_c_num: float = 1.0,
    g_c_num_obc: float | None = None,
    nu_num: float = 1.0,
    nu_num_obc: float | None = None,
) -> None:
    """FSS/collapse-style panels for transverse susceptibility chi_x data."""
    from matplotlib.lines import Line2D
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True,
                             sharey=True)
    panels = [
        (axes[0], "PBC", obs_pbc),
        (axes[1], "OBC", obs_obc),
    ]
    plotted_L_methods: dict[int, set[int]] = {}
    plotted_methods: set[int] = set()
    if g_c_num_obc is None:
        g_c_num_obc = g_c_num
    if nu_num_obc is None:
        nu_num_obc = nu_num

    for ax, bc_label, dataset in panels:
        g_ref = g_c_num if bc_label == "PBC" else g_c_num_obc
        nu_ref = nu_num if bc_label == "PBC" else nu_num_obc
        colors = colors_for_bc(bc_label)
        has_data = False
        has_ed = False
        has_lz = False
        axins = inset_axes(
            ax,
            width="38%",
            height="38%",
            loc="lower center",
            bbox_to_anchor=(-0.03, 0.05, 1.0, 1.0),
            bbox_transform=ax.transAxes,
            borderpad=1.1,
        )
        zoom_y_vals = []
        zoom_has_data = False

        for L in sorted(dataset.keys()):
            if int(L) < FINAL_LMIN_DEFAULT:
                continue
            g_arr, od = dataset[L]
            if od.shape[1] <= COL["chi_perp"]:
                continue
            chi = od[:, COL["chi_perp"]]
            m = np.isfinite(g_arr) & np.isfinite(chi)
            if np.count_nonzero(m) < 4 or not np.isfinite(g_ref):
                continue
            g_use = g_arr[m]
            chi_use = chi[m]
            order = np.argsort(g_use)
            g_use = g_use[order]
            chi_use = chi_use[order]
            g_use, unique_idx = np.unique(g_use, return_index=True)
            chi_use = chi_use[unique_idx]
            if g_use.size < 4 or g_ref < float(np.min(g_use)) or g_ref > float(np.max(g_use)):
                continue
            try:
                chi_pc = float(CubicSpline(g_use, chi_use)(g_ref))
            except (ValueError, FloatingPointError):
                continue

            method = 0 if int(L) <= ED_SIZES[-1] else 1
            ls = "-" if method == 0 else "--"
            has_ed = has_ed or method == 0
            has_lz = has_lz or method == 1
            label = rf"$L={L}$"

            x = (g_arr[m] - g_ref) * (float(L) ** (1.0 / nu_ref))
            y = g_arr[m] * chi[m] - g_ref * chi_pc
            keep = np.isfinite(x) & np.isfinite(y)
            keep &= (x >= -4.0) & (x <= 4.0)
            x = x[keep]
            y = y[keep]
            if x.size < 1:
                continue

            ax.plot(x, y, color=colors.get(L, "gray"), ls=ls, lw=1.35, label=label)
            _register_external_legend_item(plotted_L_methods, plotted_methods, int(L), method)
            mz = (x >= -0.2) & (x <= 0.2)
            if np.count_nonzero(mz) >= 2:
                axins.plot(x[mz], y[mz], color=colors.get(L, "gray"), ls=ls, lw=1.05)
                zoom_y_vals.extend(y[mz].tolist())
                zoom_has_data = True
            has_data = True

        ax.axvline(0.0, color="gray", ls=":", lw=1.2, alpha=0.7)
        ax.axhline(0.0, color="gray", ls=":", lw=1.0, alpha=0.55)
        ax.set_xlim(-4, 4)
        ax.set_xlabel(r"$(g-g_{pc})L^{1/\nu}$")
        if ax is axes[0]:
            ax.set_ylabel(r"$g\chi_x-(g\chi_x)^{*}$")
        ax.set_title(bc_label, loc="right", fontsize=TITLE_FONTSIZE)

        ax.tick_params(direction="in", which="both", top=True, right=True, labelsize=TICK_FONTSIZE)
        apply_grid(ax)
        if not has_data:
            ax.text(0.5, 0.5, "No finite data",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=TEXT_FONTSIZE, color="gray")
            axins.set_visible(False)
            continue
        if zoom_has_data:
            axins.axvline(0.0, color="gray", ls=":", lw=1.0, alpha=0.65)
            axins.axhline(0.0, color="gray", ls=":", lw=0.9, alpha=0.55)
            axins.set_xlim(-0.2, 0.2)
            zoom_y = np.asarray(zoom_y_vals, dtype=float)
            zoom_y = zoom_y[np.isfinite(zoom_y)]
            if zoom_y.size:
                ylo = float(np.min(zoom_y))
                yhi = float(np.max(zoom_y))
                if yhi > ylo:
                    pad = 0.08 * (yhi - ylo)
                    axins.set_ylim(ylo - pad, yhi + pad)
            axins.tick_params(direction="in", which="both", top=True, right=True,
                              labelsize=TICK_FONTSIZE)
            axins.yaxis.tick_right()
            axins.tick_params(axis="y", labelleft=False, labelright=True)
            apply_grid(axins)
        else:
            axins.set_visible(False)
    _external_size_method_legend(fig, plotted_L_methods, plotted_methods)
    save_fig(fig, "chi_x_fss.pdf")


def plot_fss_all_temporary(
    obs_pbc,
    obs_obc,
    chiz_pbc,
    chiz_obc,
    g_c_num: float,
    nu_num: float,
    beta_over_nu: float,
    gamma_over_nu: float,
    *,
    g_c_num_obc: float | None = None,
    nu_num_obc: float | None = None,
    beta_over_nu_obc: float | None = None,
    gamma_over_nu_obc: float | None = None,
) -> None:
    """Combined FSS collapse overview with one observable per column."""
    def _finite_or(value: float | None, fallback: float) -> float:
        return float(value) if value is not None and np.isfinite(value) else float(fallback)

    g_c_num_obc = _finite_or(g_c_num_obc, g_c_num)
    nu_num = _finite_or(nu_num, NU)
    nu_num_obc = _finite_or(nu_num_obc, nu_num)
    beta_over_nu = _finite_or(beta_over_nu, BETA / NU)
    beta_over_nu_obc = _finite_or(beta_over_nu_obc, beta_over_nu)
    gamma_over_nu = _finite_or(gamma_over_nu, GAMMA / NU)
    gamma_over_nu_obc = _finite_or(gamma_over_nu_obc, gamma_over_nu)

    axis_label_fontsize = AXIS_LABEL_FONTSIZE + 1
    tick_fontsize = TICK_FONTSIZE + 1
    title_fontsize = TITLE_FONTSIZE + 1
    text_fontsize = TEXT_FONTSIZE + 1

    fig, axes = plt.subplots(2, 4, figsize=(17.0, 8.0), constrained_layout=True)
    plotted_L_methods: dict[int, set[int]] = {}
    plotted_methods: set[int] = set()
    bc_specs = [
        ("PBC", obs_pbc, chiz_pbc, g_c_num, nu_num, beta_over_nu, gamma_over_nu),
        ("OBC", obs_obc, chiz_obc, g_c_num_obc, nu_num_obc, beta_over_nu_obc, gamma_over_nu_obc),
    ]
    psi_panels = [
        (0, COL["psi_tilde"], r"$\tilde{\Psi}\,L^{\beta/\nu}$", False),
        (1, COL["psi_bar"], r"$\bar{\Psi}\,L^{\beta/\nu}$", True),
    ]

    def _draw_psi_panel(
        ax: plt.Axes,
        bc_label: str,
        obs_data: dict,
        obs_col: int,
        ylabel: str,
        g_ref: float,
        nu_ref: float,
        beta_ref: float,
        inset_upper_left: bool,
    ) -> None:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        colors = colors_for_bc(bc_label)
        Ls = sorted(obs_data.keys())
        inset_loc = "upper left" if inset_upper_left else "upper right"
        inset_anchor = (
            (0.06, 0.0, 1.0, 1.0)
            if inset_upper_left
            else (0.0, 0.0, 1.0, 1.0)
        )
        axins = inset_axes(
            ax,
            width="34%",
            height="34%",
            loc=inset_loc,
            bbox_to_anchor=inset_anchor,
            bbox_transform=ax.transAxes,
            borderpad=1.0,
        )
        zoom_y_vals = []
        for L in Ls:
            if int(L) < 10:
                continue
            c = colors.get(L, "gray")
            method = 0 if int(L) in ED_SIZES else 1
            ls = "-" if method == 0 else "--"
            g_arr, od = obs_data[L]
            if od.shape[1] <= obs_col:
                continue
            x_res = (g_arr - g_ref) * (L ** (1.0 / nu_ref))
            y_res = od[:, obs_col] * (L ** beta_ref)
            m = np.isfinite(x_res) & np.isfinite(y_res) & (y_res > 0.0)
            if np.count_nonzero(m) < 2:
                continue
            ax.plot(x_res[m], y_res[m], color=c, ls=ls, lw=1.2,
                    label=rf"$L={L}$")
            _register_external_legend_item(plotted_L_methods, plotted_methods, int(L), method)
            mz = m & (x_res >= -0.2) & (x_res <= 0.2)
            if np.count_nonzero(mz) >= 2:
                axins.plot(x_res[mz], y_res[mz], color=c, ls=ls, lw=1.0)
                zoom_y_vals.extend(y_res[mz].tolist())

        ax.axvline(0.0, color="gray", ls=":", alpha=0.5)
        ax.set_xlim(-4, 4)
        ax.set_xlabel(r"$(g - g_{pc})\,L^{1/\nu}$", fontsize=axis_label_fontsize)
        ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)
        ax.tick_params(direction="in", which="both", top=True, right=True,
                       labelsize=tick_fontsize)
        apply_grid(ax)
        ax.set_title(bc_label, loc="right", fontsize=title_fontsize)
        axins.axvline(0.0, color="gray", ls=":", lw=1.0, alpha=0.6)
        axins.set_xlim(-0.2, 0.2)
        zoom_y = np.asarray(zoom_y_vals, dtype=float)
        zoom_y = zoom_y[np.isfinite(zoom_y)]
        if zoom_y.size:
            ylo = float(np.min(zoom_y))
            yhi = float(np.max(zoom_y))
            if yhi > ylo:
                pad = 0.08 * (yhi - ylo)
                axins.set_ylim(ylo - pad, yhi + pad)
        apply_two_decimal_y_ticks(axins)
        axins.tick_params(direction="in", which="both", top=True, right=True,
                          labelsize=tick_fontsize)
        apply_grid(axins)

    def _draw_chiz_panel(
        ax: plt.Axes,
        bc_label: str,
        dataset: dict,
        g_ref: float,
        nu_ref: float,
        gamma_ref: float,
        hide_lowest_inset_ytick: bool = False,
    ) -> None:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        colors = colors_for_bc(bc_label)
        has_data = False
        axins = inset_axes(ax, width="38%", height="38%", loc="upper right",
                           borderpad=1.1)
        zoom_y_vals = []
        zoom_has_data = False

        for L in sorted(dataset.keys()):
            if int(L) < FINAL_LMIN_DEFAULT:
                continue
            item = dataset[L]
            g_arr = np.asarray(item["g"], dtype=float)
            chi = np.asarray(item["chi"], dtype=float)
            m = np.isfinite(g_arr) & np.isfinite(chi) & (chi > 0.0)
            if np.count_nonzero(m) < 1:
                continue
            method = int(item.get("method", 0 if int(L) <= ED_SIZES[-1] else 1))
            complete = bool(item.get("complete", True))
            ls = "-" if method == 0 else "--"
            label = rf"$L={L}$"
            if not complete:
                ls = ":"
                label = rf"$L={L}$ partial {item.get('n_rows', len(g_arr))}/{item.get('n_expected', '?')}"

            x = (g_arr[m] - g_ref) * (float(L) ** (1.0 / nu_ref))
            y = chi[m] * (float(L) ** (-gamma_ref))
            keep = np.isfinite(x) & np.isfinite(y) & (y > 0.0)
            keep &= (x >= -4.0) & (x <= 4.0)
            x = x[keep]
            y = y[keep]
            if x.size < 1:
                continue

            ax.plot(x, y, color=colors.get(L, "gray"), ls=ls, lw=1.35, label=label)
            _register_external_legend_item(plotted_L_methods, plotted_methods, int(L), method)
            mz = (x >= -0.2) & (x <= 0.2)
            if np.count_nonzero(mz) >= 2:
                axins.plot(x[mz], y[mz], color=colors.get(L, "gray"), ls=ls, lw=1.05)
                zoom_y_vals.extend(y[mz].tolist())
                zoom_has_data = True
            has_data = True

        ax.axvline(0.0, color="gray", ls=":", lw=1.2, alpha=0.7)
        ax.set_xlim(-4, 4)
        ax.set_yscale("log")
        apply_plain_decimal_y_ticks(ax)
        ax.set_xlabel(r"$(g-g_{pc})L^{1/\nu}$", fontsize=axis_label_fontsize)
        ax.set_ylabel(r"$\chi_z L^{-\gamma/\nu}$", fontsize=axis_label_fontsize)
        ax.set_title(bc_label, loc="right", fontsize=title_fontsize)
        ax.tick_params(direction="in", which="both", top=True, right=True, labelsize=tick_fontsize)
        apply_grid(ax)
        if not has_data:
            ax.text(0.5, 0.5, "No finite FD data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=text_fontsize,
                    color="gray")
            axins.set_visible(False)
            return
        if zoom_has_data:
            axins.axvline(0.0, color="gray", ls=":", lw=1.0, alpha=0.65)
            axins.set_xlim(-0.2, 0.2)
            axins.set_yscale("log")
            apply_plain_decimal_y_ticks(axins)
            zoom_y = np.asarray(zoom_y_vals, dtype=float)
            zoom_y = zoom_y[np.isfinite(zoom_y) & (zoom_y > 0.0)]
            if zoom_y.size:
                ylo = float(np.min(zoom_y))
                yhi = float(np.max(zoom_y))
                if yhi > ylo:
                    axins.set_ylim(ylo / 1.08, yhi * 1.08)
            apply_two_decimal_y_ticks(axins)
            axins.tick_params(direction="in", which="both", top=True, right=True,
                              labelsize=tick_fontsize)
            if hide_lowest_inset_ytick:
                from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

                axins.yaxis.set_major_locator(FixedLocator([1.0]))
                axins.yaxis.set_major_formatter(FixedFormatter(["1.00"]))
                axins.yaxis.set_minor_formatter(NullFormatter())
            apply_grid(axins)
        else:
            axins.set_visible(False)

    def _draw_chiperp_panel(
        ax: plt.Axes,
        bc_label: str,
        obs_data: dict,
        g_ref: float,
        nu_ref: float,
    ) -> None:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        colors = colors_for_bc(bc_label)
        has_data = False
        axins = inset_axes(
            ax,
            width="34%",
            height="34%",
            loc="lower center",
            bbox_to_anchor=(0.0, 0.06, 1.0, 1.0),
            bbox_transform=ax.transAxes,
            borderpad=1.0,
        )
        zoom_y_vals = []
        zoom_has_data = False

        for L in sorted(obs_data.keys()):
            if int(L) < FINAL_LMIN_DEFAULT:
                continue
            g_arr, od = obs_data[L]
            if od.shape[1] <= COL["chi_perp"]:
                continue
            chi = od[:, COL["chi_perp"]]
            m = np.isfinite(g_arr) & np.isfinite(chi)
            if np.count_nonzero(m) < 4 or not np.isfinite(g_ref):
                continue
            g_use = g_arr[m]
            chi_use = chi[m]
            order = np.argsort(g_use)
            g_use = g_use[order]
            chi_use = chi_use[order]
            g_use, unique_idx = np.unique(g_use, return_index=True)
            chi_use = chi_use[unique_idx]
            if g_use.size < 4 or g_ref < float(np.min(g_use)) or g_ref > float(np.max(g_use)):
                continue
            try:
                chi_pc = float(CubicSpline(g_use, chi_use)(g_ref))
            except (ValueError, FloatingPointError):
                continue
            method = 0 if int(L) in ED_SIZES else 1
            ls = "-" if method == 0 else "--"

            x = (g_arr[m] - g_ref) * (float(L) ** (1.0 / nu_ref))
            y = g_arr[m] * chi[m] - g_ref * chi_pc
            keep = np.isfinite(x) & np.isfinite(y)
            keep &= (x >= -4.0) & (x <= 4.0)
            x = x[keep]
            y = y[keep]
            if x.size < 1:
                continue

            ax.plot(x, y, color=colors.get(L, "gray"), ls=ls, lw=1.35,
                    label=rf"$L={L}$")
            _register_external_legend_item(plotted_L_methods, plotted_methods, int(L), method)
            mz = (x >= -0.2) & (x <= 0.2)
            if np.count_nonzero(mz) >= 2:
                axins.plot(x[mz], y[mz], color=colors.get(L, "gray"),
                           ls=ls, lw=1.05)
                zoom_y_vals.extend(y[mz].tolist())
                zoom_has_data = True
            has_data = True

        ax.axvline(0.0, color="gray", ls=":", lw=1.2, alpha=0.7)
        ax.axhline(0.0, color="gray", ls=":", lw=1.0, alpha=0.55)
        ax.set_xlim(-4, 4)
        ax.set_xlabel(r"$(g-g_{pc})L^{1/\nu}$", fontsize=axis_label_fontsize)
        ax.set_ylabel(r"$g\chi_x-(g\chi_x)^{*}$",
                      fontsize=axis_label_fontsize)
        ax.set_title(bc_label, loc="right", fontsize=title_fontsize)
        ax.tick_params(direction="in", which="both", top=True, right=True,
                       labelsize=tick_fontsize)
        apply_grid(ax)
        if not has_data:
            ax.text(0.5, 0.5, "No finite data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=text_fontsize,
                    color="gray")
            axins.set_visible(False)
            return
        if zoom_has_data:
            axins.axvline(0.0, color="gray", ls=":", lw=1.0, alpha=0.65)
            axins.axhline(0.0, color="gray", ls=":", lw=0.9, alpha=0.55)
            axins.set_xlim(-0.2, 0.2)
            zoom_y = np.asarray(zoom_y_vals, dtype=float)
            zoom_y = zoom_y[np.isfinite(zoom_y)]
            if zoom_y.size:
                ylo = float(np.min(zoom_y))
                yhi = float(np.max(zoom_y))
                if yhi > ylo:
                    pad = 0.08 * (yhi - ylo)
                    axins.set_ylim(ylo - pad, yhi + pad)
            axins.tick_params(direction="in", which="both", top=True, right=True,
                              labelsize=tick_fontsize)
            apply_grid(axins)
        else:
            axins.set_visible(False)
    for row_idx, (bc_label, obs_data, chiz_data, g_ref, nu_ref, beta_ref, gamma_ref) in enumerate(bc_specs):
        for col_idx, obs_col, ylabel, _inset_upper_left in psi_panels:
            _draw_psi_panel(
                axes[row_idx, col_idx],
                bc_label,
                obs_data,
                obs_col,
                ylabel,
                g_ref,
                nu_ref,
                beta_ref,
                inset_upper_left=False,
            )
        _draw_chiz_panel(
            axes[row_idx, 2],
            bc_label,
            chiz_data,
            g_ref,
            nu_ref,
            gamma_ref,
            hide_lowest_inset_ytick=True,
        )
        _draw_chiperp_panel(
            axes[row_idx, 3],
            bc_label,
            obs_data,
            g_ref,
            nu_ref,
        )

    for col_idx in range(4):
        axes[0, col_idx].set_xlabel("")
        axes[0, col_idx].tick_params(labelbottom=False)

    chi_ylims = [axes[row_idx, 2].get_ylim() for row_idx in range(2)]
    chi_ylo = min(lo for lo, _hi in chi_ylims)
    chi_yhi = max(hi for _lo, hi in chi_ylims)
    for row_idx in range(2):
        axes[row_idx, 2].set_ylim(chi_ylo, chi_yhi)

    chiperp_ylims = [axes[row_idx, 3].get_ylim() for row_idx in range(2)]
    chiperp_ylo = min(lo for lo, _hi in chiperp_ylims)
    chiperp_yhi = max(hi for _lo, hi in chiperp_ylims)
    for row_idx in range(2):
        axes[row_idx, 3].set_ylim(chiperp_ylo, chiperp_yhi)

    _external_size_method_legend(fig, plotted_L_methods, plotted_methods, y_anchor=-0.10)
    save_fig(fig, "fss_all.pdf")


# ---------------------------------------------------------------------------
# Gap data loaders
# ---------------------------------------------------------------------------
def load_gap_file(path: Path) -> np.ndarray | None:
    """Load a gap_*.dat file. Returns (N, >=7) array or None."""
    if not path.exists():
        return None
    try:
        d = np.atleast_2d(np.loadtxt(path, comments="#"))
        if d.ndim < 2 or d.shape[1] < 7:
            return None
        return d
    except Exception as e:
        print(f"  [WARN] {path.name}: {e}")
        return None


def gather_gap_data(L_list: list[int], pbc: bool = True, lanczos: bool = False) -> dict[int, np.ndarray]:
    """
    Load gap data for all L in L_list, trying the requested backend first
    and falling back to the other one if available. Returns {L: array}.
    """
    data = {}
    bc = "" if pbc else "_obc"
    data_dir = data_dir_for_bc(pbc)

    for L in L_list:
        candidates = (
            [("Lanczos", f"gap_lz{bc}_L{L:02d}.dat"), ("full-ED", f"gap{bc}_L{L:02d}.dat")]
            if lanczos
            else [("full-ED", f"gap{bc}_L{L:02d}.dat"), ("Lanczos", f"gap_lz{bc}_L{L:02d}.dat")]
        )
        d = None
        src = ""
        for src_try, fname in candidates:
            d = load_gap_file(data_dir / fname)
            if d is not None:
                src = src_try
                break
        if d is not None:
            data[L] = d
            print(f"  L={L:2d} [{src}]: {d.shape[0]} g-points")
        else:
            print(f"  L={L:2d}: no data found")
    return data


# ---------------------------------------------------------------------------
# Deterministic OLS FSS pipeline (ED data: no statistical resampling)
# ---------------------------------------------------------------------------
def _power_model(L, A, x):
    return A * np.asarray(L, dtype=float) ** x


def _inv_power_model(L, A, x):
    return A * np.asarray(L, dtype=float) ** (-x)


def _curve_fit_deterministic(*args, **kwargs):
    """Run curve_fit as an optimizer; its covariance output is intentionally ignored."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        return curve_fit(*args, **kwargs)


def fit_powerlaw_ols(
    L_arr: np.ndarray,
    y_arr: np.ndarray,
    *,
    inverse: bool = False,
) -> dict:
    """Leading-only deterministic fit."""
    L_arr = np.asarray(L_arr, dtype=float)
    y_arr = np.asarray(y_arr, dtype=float)
    m = np.isfinite(L_arr) & np.isfinite(y_arr) & (L_arr > 0.0) & (y_arr > 0.0)
    L_use = L_arr[m]
    y_use = y_arr[m]
    if L_use.size:
        idx = np.argsort(L_use)
        L_use = L_use[idx]
        y_use = y_use[idx]

    out = dict(
        A=np.nan, exponent=np.nan, resid_rms=np.nan, dof=0, L=L_use, y=y_use,
    )
    if L_use.size < 3:
        return out

    model = _inv_power_model if inverse else _power_model
    sign = -1.0 if inverse else 1.0
    coeff = np.polyfit(np.log(L_use), np.log(y_use), 1)
    p0 = [float(np.exp(coeff[1])), float(sign * coeff[0])]

    try:
        popt, _pcov = _curve_fit_deterministic(
            model, L_use, y_use, p0=p0, maxfev=10000,
        )
    except Exception:
        return out

    y_fit = model(L_use, *popt)
    resid = y_use - y_fit
    dof = max(0, int(L_use.size - len(popt)))

    out.update(
        A=float(popt[0]),
        exponent=float(popt[1]),
        resid_rms=float(np.sqrt(np.mean(resid ** 2))) if resid.size else np.nan,
        dof=dof,
    )
    return out


def _lmin_drift(sweep: list[dict], final_lmin: int) -> float:
    final = next((r for r in sweep if int(r["Lmin"]) == int(final_lmin)), None)
    if final is None or not np.isfinite(final["value"]):
        return np.nan
    vals = [
        abs(r["value"] - final["value"])
        for r in sweep
        if int(r["Lmin"]) >= int(final_lmin) and np.isfinite(r["value"])
    ]
    return float(max(vals)) if vals else np.nan


def _sweep_powerlaw(
    L_arr: np.ndarray,
    y_arr: np.ndarray,
    *,
    inverse: bool = False,
    lmins: list[int] = LMIN_SWEEP,
) -> list[dict]:
    rows = []
    L_arr = np.asarray(L_arr, dtype=float)
    y_arr = np.asarray(y_arr, dtype=float)
    for Lmin in lmins:
        m = L_arr >= float(Lmin)
        fit = fit_powerlaw_ols(L_arr[m], y_arr[m], inverse=inverse)
        rows.append(dict(
            Lmin=int(Lmin),
            value=fit["exponent"],
            A=fit["A"],
            resid_rms=fit["resid_rms"],
            dof=fit["dof"],
            n=int(np.count_nonzero(m)),
        ))
    return rows


def _powerlaw_subleading_omega(bc: str, use_effective_length: bool, omega: float | None) -> float:
    """Default fixed correction exponent for subleading power-law fits."""
    if omega is not None:
        return float(omega)
    bc_key = str(bc).lower()
    if use_effective_length:
        return float(GAP_RESIDUAL_OMEGA)
    return float(OMEGA_PBC if bc_key == "pbc" else OMEGA_OBC)


def _nan_powerlaw_subleading_row(
    Lmin: int,
    bc: str,
    inverse: bool,
    use_effective_length: bool,
    omega_fixed: float,
    npts: int = 0,
) -> dict:
    bc_key = str(bc).lower()
    shift = OBC_LE_SHIFT if bc_key == "obc" and use_effective_length else 0.0
    return dict(
        Lmin=int(Lmin),
        npts=int(npts),
        n=int(npts),
        A=np.nan,
        exponent=np.nan,
        B=np.nan,
        omega_fixed=float(omega_fixed) if np.isfinite(omega_fixed) else np.nan,
        use_effective_length=bool(use_effective_length),
        inverse=bool(inverse),
        Le_shift_used=float(shift),
        value=np.nan,
        resid_rms=np.nan,
        dof=0,
    )


def fit_powerlaw_subleading_fixed_omega(
    L,
    y,
    bc,
    Lmin,
    *,
    inverse=False,
    use_effective_length=False,
    omega=None,
) -> dict:
    """
    Fit y(L)=A*X^(+/-x)*(1+B*X^-omega) with fixed omega.

    X is either the raw length L or the boundary-condition effective length.
    With inverse=True the reported positive exponent x corresponds to y~X^-x.
    """
    bc_key = str(bc).lower()
    omega_fixed = _powerlaw_subleading_omega(bc_key, use_effective_length, omega)
    L_arr = np.asarray(L, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    m = (
        np.isfinite(L_arr)
        & np.isfinite(y_arr)
        & (L_arr > 0.0)
        & (y_arr > 0.0)
        & (L_arr >= float(Lmin))
    )
    L_use = L_arr[m]
    y_use = y_arr[m]
    if L_use.size:
        idx = np.argsort(L_use)
        L_use = L_use[idx]
        y_use = y_use[idx]
    if L_use.size < 4:
        return _nan_powerlaw_subleading_row(
            Lmin, bc_key, inverse, use_effective_length, omega_fixed, L_use.size
        )

    try:
        X = effective_length(L_use, bc_key) if use_effective_length else np.asarray(L_use, dtype=float)
    except ValueError:
        return _nan_powerlaw_subleading_row(
            Lmin, bc_key, inverse, use_effective_length, omega_fixed, L_use.size
        )

    ok = np.isfinite(X) & (X > 0.0) & np.isfinite(y_use) & (y_use > 0.0)
    X = X[ok]
    y_use = y_use[ok]
    if X.size < 4:
        return _nan_powerlaw_subleading_row(
            Lmin, bc_key, inverse, use_effective_length, omega_fixed, X.size
        )

    lead = fit_powerlaw_ols(X, y_use, inverse=inverse)
    A0 = lead.get("A", np.nan)
    x0 = lead.get("exponent", np.nan)
    if not (np.isfinite(A0) and A0 > 0.0 and np.isfinite(x0)):
        coeff = np.polyfit(np.log(X), np.log(y_use), 1)
        A0 = float(np.exp(coeff[1]))
        x0 = float(-coeff[0] if inverse else coeff[0])
    if not (np.isfinite(A0) and A0 > 0.0 and np.isfinite(x0)):
        return _nan_powerlaw_subleading_row(
            Lmin, bc_key, inverse, use_effective_length, omega_fixed, X.size
        )

    sign = -1.0 if inverse else 1.0

    def _model(X_vals, A, exponent, B):
        X_vals = np.asarray(X_vals, dtype=float)
        return A * X_vals ** (sign * exponent) * (1.0 + B * X_vals ** (-omega_fixed))

    try:
        popt, _pcov = _curve_fit_deterministic(
            _model,
            X,
            y_use,
            p0=[A0, x0, 0.0],
            maxfev=50000,
        )
    except (RuntimeError, ValueError, FloatingPointError):
        return _nan_powerlaw_subleading_row(
            Lmin, bc_key, inverse, use_effective_length, omega_fixed, X.size
        )

    A, exponent, B = (float(popt[0]), float(popt[1]), float(popt[2]))
    y_fit = _model(X, A, exponent, B)
    resid = y_use - y_fit
    shift = OBC_LE_SHIFT if bc_key == "obc" and use_effective_length else 0.0
    return dict(
        Lmin=int(Lmin),
        npts=int(X.size),
        n=int(X.size),
        A=A,
        exponent=exponent,
        B=B,
        omega_fixed=float(omega_fixed),
        use_effective_length=bool(use_effective_length),
        inverse=bool(inverse),
        Le_shift_used=float(shift),
        value=exponent,
        resid_rms=float(np.sqrt(np.mean(resid ** 2))) if resid.size else np.nan,
        dof=max(0, int(X.size - 3)),
    )


def _sweep_powerlaw_subleading_fixed_omega(
    L_arr: np.ndarray,
    y_arr: np.ndarray,
    bc: str,
    *,
    inverse: bool = False,
    use_effective_length: bool = False,
    omega: float | None = None,
    lmins: list[int] = LMIN_SWEEP,
) -> list[dict]:
    rows = []
    for Lmin in lmins:
        rows.append(
            fit_powerlaw_subleading_fixed_omega(
                L_arr,
                y_arr,
                bc,
                Lmin,
                inverse=inverse,
                use_effective_length=use_effective_length,
                omega=omega,
            )
        )
    return rows


def _row_for_lmin(block: dict, Lmin: int) -> dict | None:
    sweep = block.get("sweep", []) if block else []
    row = next((r for r in sweep if int(r["Lmin"]) == int(Lmin)), None)
    if row is None:
        row = next((r for r in sweep if int(r["Lmin"]) >= int(Lmin)), None)
    if row is None and sweep:
        row = sweep[-1]
    return row


def _eval_powerlaw_fit_row(L, row: dict, bc: str, *, inverse: bool) -> np.ndarray:
    L_arr = np.asarray(L, dtype=float)
    if not row or not (np.isfinite(row.get("A", np.nan)) and np.isfinite(row.get("value", np.nan))):
        return np.full_like(L_arr, np.nan, dtype=float)
    use_effective_length = bool(row.get("use_effective_length", False))
    X = effective_length(L_arr, bc) if use_effective_length else L_arr
    sign = -1.0 if inverse else 1.0
    y_fit = row["A"] * X ** (sign * row["value"])
    B = row.get("B", np.nan)
    omega_fixed = row.get("omega_fixed", np.nan)
    if np.isfinite(B) and np.isfinite(omega_fixed):
        y_fit = y_fit * (1.0 + B * X ** (-omega_fixed))
    return y_fit


def effective_length(L, bc):
    """
    Effective length for spectral/gap-based OBC FSS.

    For PBC, Le=L and this is not a separate fitting prescription.  For the
    order parameter the pipeline deliberately uses raw L only, because
    non-RG-invariant amplitudes can retain O(1/L) corrections not absorbed by
    redefining the chain length.
    """
    bc_key = str(bc).lower()
    L_arr = np.asarray(L, dtype=float)
    if bc_key == "pbc":
        return L_arr
    if bc_key == "obc":
        return L_arr + OBC_LE_SHIFT
    raise ValueError(f"unknown boundary condition: {bc}")


def _nan_gap_z_row(
    Lmin: int,
    bc: str,
    use_effective_length: bool,
    omega_fixed: float,
    npts: int = 0,
) -> dict:
    shift = OBC_LE_SHIFT if str(bc).lower() == "obc" and use_effective_length else 0.0
    return dict(
        Lmin=int(Lmin),
        npts=int(npts),
        n=int(npts),
        A=np.nan,
        z=np.nan,
        B=np.nan,
        omega_fixed=float(omega_fixed) if np.isfinite(omega_fixed) else np.nan,
        use_effective_length=bool(use_effective_length),
        Le_shift_used=float(shift),
        value=np.nan,
        resid_rms=np.nan,
        dof=0,
    )


def fit_gap_z_subleading(
    L,
    gap,
    bc,
    Lmin,
    use_effective_length: bool = True,
    omega: float | None = None,
) -> dict:
    """
    Fixed-omega critical-gap fit.

    Final corrected fit:
        Delta0(gc,L) = A * Le^(-z) * (1 + B * Le^(-2)).

    For OBC the open-chain XY/Ising size scaling field has an O(1/L)
    correction that is absorbed at gamma=1 by Le=L+1/2; the residual
    critical-gap correction is O(Le^-2). This follows the FSS analysis of
    Campostrini, Pelissetto, and Vicari, arXiv:1401.0788.
    """
    bc_key = str(bc).lower()
    if use_effective_length:
        omega_fixed = GAP_RESIDUAL_OMEGA
    else:
        omega_fixed = (
            float(omega)
            if omega is not None
            else (GAP_RESIDUAL_OMEGA if bc_key == "pbc" else OBC_RAW_OMEGA)
        )

    L_arr = np.asarray(L, dtype=float)
    gap_arr = np.asarray(gap, dtype=float)
    m = (
        np.isfinite(L_arr)
        & np.isfinite(gap_arr)
        & (gap_arr > 0.0)
        & (L_arr >= float(Lmin))
    )
    L_use = L_arr[m]
    gap_use = gap_arr[m]
    if L_use.size:
        idx = np.argsort(L_use)
        L_use = L_use[idx]
        gap_use = gap_use[idx]
    if L_use.size < 4:
        return _nan_gap_z_row(Lmin, bc_key, use_effective_length, omega_fixed, L_use.size)

    try:
        Le = effective_length(L_use, bc_key) if use_effective_length else np.asarray(L_use, dtype=float)
    except ValueError:
        return _nan_gap_z_row(Lmin, bc_key, use_effective_length, omega_fixed, L_use.size)

    ok = np.isfinite(Le) & (Le > 0.0) & np.isfinite(gap_use) & (gap_use > 0.0)
    Le = Le[ok]
    gap_use = gap_use[ok]
    if Le.size < 4:
        return _nan_gap_z_row(Lmin, bc_key, use_effective_length, omega_fixed, Le.size)

    try:
        coeff = np.polyfit(np.log(Le), np.log(gap_use), 1)
        z0 = float(-coeff[0])
        A0 = float(np.exp(coeff[1]))
        if not (np.isfinite(A0) and A0 > 0.0 and np.isfinite(z0)):
            return _nan_gap_z_row(Lmin, bc_key, use_effective_length, omega_fixed, Le.size)

        def _model(Le_vals, A, z, B):
            Le_vals = np.asarray(Le_vals, dtype=float)
            return A * Le_vals ** (-z) * (1.0 + B * Le_vals ** (-omega_fixed))

        popt, _pcov = _curve_fit_deterministic(
            _model,
            Le,
            gap_use,
            p0=[A0, z0, 0.0],
            maxfev=50000,
        )
    except (RuntimeError, ValueError, FloatingPointError):
        return _nan_gap_z_row(Lmin, bc_key, use_effective_length, omega_fixed, Le.size)

    A, z, B = (float(popt[0]), float(popt[1]), float(popt[2]))
    y_fit = A * Le ** (-z) * (1.0 + B * Le ** (-omega_fixed))
    resid = gap_use - y_fit
    shift = OBC_LE_SHIFT if bc_key == "obc" and use_effective_length else 0.0
    return dict(
        Lmin=int(Lmin),
        npts=int(Le.size),
        n=int(Le.size),
        A=A,
        z=z,
        B=B,
        omega_fixed=float(omega_fixed),
        use_effective_length=bool(use_effective_length),
        Le_shift_used=float(shift),
        value=z,
        resid_rms=float(np.sqrt(np.mean(resid ** 2))) if resid.size else np.nan,
        dof=max(0, int(Le.size - 3)),
    )


def z_dynamic_subleading_sweep(
    L,
    gap,
    bc,
    lmins: list[int] = LMIN_SWEEP,
    use_effective_length: bool = True,
) -> list[dict]:
    rows = []
    for Lmin in lmins:
        try:
            rows.append(
                fit_gap_z_subleading(
                    L,
                    gap,
                    bc,
                    Lmin,
                    use_effective_length=use_effective_length,
                )
            )
        except (RuntimeError, ValueError, FloatingPointError):
            bc_key = str(bc).lower()
            omega = GAP_RESIDUAL_OMEGA if use_effective_length or bc_key == "pbc" else OBC_RAW_OMEGA
            rows.append(_nan_gap_z_row(Lmin, bc_key, use_effective_length, omega, 0))
    return rows


def gap_scaled_crossings_table(
    gap_data: dict[int, np.ndarray],
    *,
    bc: str = "pbc",
    use_effective_length: bool = False,
) -> list[dict]:
    """Deterministic crossings of X*Delta0 between consecutive L values."""
    rows = []
    Ls = sorted(gap_data)
    for i in range(len(Ls) - 1):
        L1, L2 = Ls[i], Ls[i + 1]
        d1, d2 = gap_data[L1], gap_data[L2]
        c1, c2 = gap_columns(d1), gap_columns(d2)
        g1 = np.asarray(c1["g_arr"], dtype=float)
        g2 = np.asarray(c2["g_arr"], dtype=float)
        scale1 = float(effective_length(np.asarray([L1], dtype=float), bc)[0]) if use_effective_length else float(L1)
        scale2 = float(effective_length(np.asarray([L2], dtype=float), bc)[0]) if use_effective_length else float(L2)
        y1 = np.asarray(c1["Delta0"], dtype=float) * scale1
        y2 = np.asarray(c2["Delta0"], dtype=float) * scale2
        m1 = np.isfinite(g1) & np.isfinite(y1)
        m2 = np.isfinite(g2) & np.isfinite(y2)
        if np.count_nonzero(m1) < 4 or np.count_nonzero(m2) < 4:
            continue

        g1, y1 = g1[m1], y1[m1]
        g2, y2 = g2[m2], y2[m2]
        f1 = interp1d(g1, y1, kind="cubic", bounds_error=False, fill_value=np.nan)
        f2 = interp1d(g2, y2, kind="cubic", bounds_error=False, fill_value=np.nan)
        glo = max(float(np.min(g1)), float(np.min(g2)), 0.7)
        ghi = min(float(np.max(g1)), float(np.max(g2)), 1.3)
        if ghi <= glo:
            continue

        grid = np.linspace(glo, ghi, 2000)
        diff = np.asarray(f1(grid) - f2(grid), dtype=float)
        ok = np.isfinite(diff)
        if np.count_nonzero(ok) < 2:
            continue
        grid_ok = grid[ok]
        diff_ok = diff[ok]
        idx = np.where(np.diff(np.sign(diff_ok)) != 0)[0]
        if idx.size == 0:
            continue
        best = idx[np.argmin(np.abs(grid_ok[idx] - G_C))]
        try:
            g_pc = float(brentq(lambda g: float(f1(g) - f2(g)),
                                grid_ok[best], grid_ok[best + 1]))
        except ValueError:
            continue
        rows.append(dict(L1=int(L1), L2=int(L2), L_eff=0.5 * (L1 + L2),
                         scale_eff=0.5 * (scale1 + scale2), g_pc=g_pc,
                         use_effective_length=bool(use_effective_length)))
    return rows


def weighted_gc_from_crossings(rows: list[dict], Lmin: int | None = None) -> float:
    vals, weights = [], []
    for r in rows:
        if Lmin is not None and r["L_eff"] < float(Lmin):
            continue
        if np.isfinite(r["g_pc"]):
            vals.append(float(r["g_pc"]))
            weights.append(float(r["L_eff"]) ** 2)
    if not vals:
        return np.nan
    vals = np.asarray(vals, dtype=float)
    weights = np.asarray(weights, dtype=float)
    return float(np.sum(vals * weights) / np.sum(weights))


def gc_sweep_from_crossings(rows: list[dict]) -> list[dict]:
    out = []
    for Lmin in LMIN_SWEEP:
        val = weighted_gc_from_crossings(rows, Lmin=Lmin)
        n = sum(1 for r in rows if r["L_eff"] >= float(Lmin))
        out.append(dict(Lmin=int(Lmin), value=val, n=n))
    return out


def fit_gpc_from_crossings(
    rows: list[dict],
    *,
    approach: str,
    omega: float | None = None,
    Lmin: int = FINAL_LMIN_DEFAULT,
) -> dict:
    """Estimate one approach pseudo-critical point from crossing data."""
    selected = [
        r for r in rows
        if float(r.get("L_eff", np.nan)) >= float(Lmin)
        and np.isfinite(r.get("g_pc", np.nan))
    ]
    if not selected:
        return dict(value=np.nan, source=approach, Lmin=int(Lmin), n=0)

    if approach == "leading" or omega is None or not np.isfinite(omega):
        value = weighted_gc_from_crossings(rows, Lmin=Lmin)
        return dict(value=float(value), source="raw_L_leading", Lmin=int(Lmin), n=len(selected))

    scale = np.asarray([r.get("scale_eff", r["L_eff"]) for r in selected], dtype=float)
    y = np.asarray([r["g_pc"] for r in selected], dtype=float)
    m = np.isfinite(scale) & (scale > 0.0) & np.isfinite(y)
    if np.count_nonzero(m) < 2:
        value = weighted_gc_from_crossings(rows, Lmin=Lmin)
        return dict(value=float(value), source=f"{approach}_weighted_fallback", Lmin=int(Lmin), n=len(selected))
    x = scale[m] ** (-float(omega))
    coeff = np.linalg.lstsq(np.column_stack([np.ones_like(x), x]), y[m], rcond=None)[0]
    return dict(value=float(coeff[0]), source=approach, Lmin=int(Lmin), n=int(np.count_nonzero(m)))


def gap_derivative_table(gap_data: dict[int, np.ndarray], g_pc: float) -> dict[str, np.ndarray]:
    L_vals, dlog_vals, gap_vals = [], [], []
    for L in sorted(gap_data):
        gd = gap_data[L]
        cols = gap_columns(gd)
        g = np.asarray(cols["g_arr"], dtype=float)
        gap = np.asarray(cols["Delta0"], dtype=float)
        m = np.isfinite(g) & np.isfinite(gap) & (gap > 0.0)
        if np.count_nonzero(m) < 4:
            continue
        g_use = g[m]
        gap_use = gap[m]
        if g_pc < np.min(g_use) or g_pc > np.max(g_use):
            continue
        spl = CubicSpline(g_use, gap_use)
        gap_pc = float(spl(g_pc))
        deriv = float(spl(g_pc, 1))
        if np.isfinite(gap_pc) and gap_pc > 0.0 and np.isfinite(deriv):
            L_vals.append(float(L))
            gap_vals.append(gap_pc)
            dlog_vals.append(abs(deriv / gap_pc))
    return dict(L=np.asarray(L_vals), gap=np.asarray(gap_vals),
                dlog=np.asarray(dlog_vals))


def chi_peak_tables(obs_data: dict[int, tuple]) -> dict[str, np.ndarray]:
    """Return an empty peak table; chi_z is read from finite-difference data."""
    _ = obs_data
    return dict(
        L=np.asarray([], dtype=float),
        g_peak=np.asarray([], dtype=float),
        chi_max=np.asarray([], dtype=float),
    )


def psifss_tables(obs_data: dict[int, tuple], g_c_num: float) -> dict[str, np.ndarray]:
    """Extract psi_tilde(gc,L) and psi_bar(gc,L) for all L."""
    L_vals, psi_t_vals, psi_b_vals = [], [], []
    for L in sorted(obs_data):
        g_arr, od = obs_data[L]
        if od.shape[1] <= max(COL["psi_tilde"], COL["psi_bar"]):
            continue
        vals = {}
        for name, col in (("psiT", COL["psi_tilde"]), ("psiB", COL["psi_bar"])):
            col_data = np.asarray(od[:, col], dtype=float)
            m = np.isfinite(g_arr) & np.isfinite(col_data)
            if np.count_nonzero(m) < 4:
                break
            g_use = np.asarray(g_arr[m], dtype=float)
            y_use = np.asarray(col_data[m], dtype=float)
            if g_c_num < np.min(g_use) or g_c_num > np.max(g_use):
                break
            spl = CubicSpline(g_use, y_use)
            vals[name] = float(spl(g_c_num))
        else:
            psi_t = vals.get("psiT", np.nan)
            psi_b = vals.get("psiB", np.nan)
            if (np.isfinite(psi_t) and psi_t > 0.0 and
                    np.isfinite(psi_b) and psi_b > 0.0):
                L_vals.append(float(L))
                psi_t_vals.append(float(psi_t))
                psi_b_vals.append(float(psi_b))
    return dict(
        L=np.asarray(L_vals),
        psiT=np.asarray(psi_t_vals),
        psiB=np.asarray(psi_b_vals),
    )


def _select_final_lmin(primary: dict) -> int:
    """Smallest conservative cutoff where primary exponent sweeps stabilize."""
    valid_lmins = []
    for Lmin in LMIN_SWEEP:
        ok = True
        for key in ("nu_inv", "beta_over_nu", "beta_over_nu_bar"):
            sweep = primary.get(key, {}).get("sweep", [])
            row = next((r for r in sweep if int(r["Lmin"]) == int(Lmin)), None)
            if row is None or not np.isfinite(row["value"]):
                ok = False
                break
        if ok:
            valid_lmins.append(int(Lmin))

    for Lmin in LMIN_SWEEP:
        if Lmin < FINAL_LMIN_DEFAULT:
            continue
        if int(Lmin) not in valid_lmins:
            continue
        ok_all = True
        for key in ("nu_inv", "beta_over_nu", "beta_over_nu_bar"):
            sweep = primary.get(key, {}).get("sweep", [])
            if not sweep:
                ok_all = False
                break
            drift = _lmin_drift(sweep, Lmin)
            final = next((r for r in sweep if int(r["Lmin"]) == int(Lmin)), None)
            if final is None or not np.isfinite(final["value"]) or not np.isfinite(drift):
                ok_all = False
                break
            hi = final["value"] + drift
            lo = final["value"] - drift
            for r in sweep:
                if int(r["Lmin"]) >= int(Lmin) and np.isfinite(r["value"]):
                    if r["value"] < lo - 1e-12 or r["value"] > hi + 1e-12:
                        ok_all = False
                        break
            if not ok_all:
                break
        if ok_all:
            return int(Lmin)
    return max(valid_lmins) if valid_lmins else FINAL_LMIN_DEFAULT


def make_gpc_specs(
    bc_label: str,
    raw_crossings: list[dict],
    effective_crossings: list[dict] | None = None,
) -> dict:
    """Return approach-specific pseudo-critical points."""
    bc_key = str(bc_label).lower()
    raw_omega = OMEGA_PBC if bc_key == "pbc" else OBC_RAW_OMEGA
    leading = fit_gpc_from_crossings(raw_crossings, approach="leading", Lmin=FINAL_LMIN_DEFAULT)
    raw_sub = fit_gpc_from_crossings(
        raw_crossings,
        approach="subleading" if bc_key == "pbc" else "raw_subleading_diagnostic",
        omega=raw_omega,
        Lmin=FINAL_LMIN_DEFAULT,
    )
    specs = {
        "leading": {
            **leading,
            "approach": "leading",
            "source": "raw_L_leading",
            "crossing_scale": "raw_L",
        },
    }
    if bc_key == "pbc":
        specs["raw_subleading"] = {
            **raw_sub,
            "approach": "raw_subleading",
            "source": "raw_L_subleading",
            "crossing_scale": "raw_L",
        }
        return specs

    if bc_key == "obc":
        specs["raw_subleading_diagnostic"] = {
            **raw_sub,
            "approach": "raw_subleading_diagnostic",
            "source": "raw_L_subleading",
            "crossing_scale": "raw_L",
        }
        eff_rows = effective_crossings or raw_crossings
        mixed = fit_gpc_from_crossings(
            eff_rows,
            approach="mixed_subleading",
            omega=GAP_RESIDUAL_OMEGA,
            Lmin=FINAL_LMIN_DEFAULT,
        )
        specs["mixed_subleading"] = {
            **mixed,
            "approach": "mixed_subleading",
            "source": "Le_subleading",
            "crossing_scale": "Le=L+1/2",
        }
        g_raw = specs["raw_subleading_diagnostic"].get("value", np.nan)
        g_mixed = specs["mixed_subleading"].get("value", np.nan)
        if np.isfinite(g_raw) and np.isfinite(g_mixed) and abs(g_raw - g_mixed) <= 1e-10:
            warnings.warn(
                "OBC raw_subleading_diagnostic and mixed_subleading g_pc are equal within 1e-10.",
                RuntimeWarning,
            )
    return specs


def _attach_approach_metadata(block: dict, approach_data: dict) -> dict:
    block["g_pc"] = float(approach_data.get("g_pc", np.nan))
    block["gpc_source"] = str(approach_data.get("gpc_source", ""))
    block["gpc_approach"] = str(approach_data.get("approach", ""))
    return block


def public_approach_label(bc_label: str, approach: str) -> str:
    """Human-readable approach names used in generated outputs."""
    bc_key = str(bc_label).upper()
    if bc_key == "PBC" and approach == "raw_subleading":
        return "subleading"
    if bc_key == "OBC" and approach == "raw_subleading":
        return "raw_subleading_diagnostic"
    if bc_key == "OBC" and approach == "effective_length":
        return "mixed_subleading"
    return str(approach)


def approach_block_key(approach: str, block_name: str) -> str:
    """Return the primary-dictionary key for a block inside one approach."""
    return f"{block_name}_{approach}"


def build_primary_fss(
    gap_data: dict[int, np.ndarray],
    obs_data: dict[int, tuple],
    bc_label: str,
    raw_crossings: list[dict],
    gpc_specs: dict,
) -> dict:
    bc_key = str(bc_label).lower()
    is_pbc = bc_key == "pbc"
    raw_omega = OMEGA_PBC if is_pbc else OBC_RAW_OMEGA
    chi_peaks = chi_peak_tables(obs_data)
    channels: dict[str, dict] = {}

    for approach, spec in gpc_specs.items():
        g_pc = float(spec.get("value", np.nan))
        deriv = gap_derivative_table(gap_data, g_pc)
        psi_tables = psifss_tables(obs_data, g_pc)
        channel = dict(
            approach=approach,
            g_pc=g_pc,
            gpc_source=str(spec.get("source", approach)),
            gpc_crossing_scale=str(spec.get("crossing_scale", "raw_L")),
            gpc_lmin=int(spec.get("Lmin", FINAL_LMIN_DEFAULT)),
            gpc_n=int(spec.get("n", 0)),
            gap_derivative=deriv,
            psi=psi_tables,
        )

        if approach == "leading":
            channel["nu_inv"] = dict(sweep=_sweep_powerlaw(deriv["L"], deriv["dlog"], inverse=False))
            channel["z_dynamic"] = dict(sweep=_sweep_powerlaw(deriv["L"], deriv["gap"], inverse=True))
            channel["beta_over_nu"] = dict(
                sweep=_sweep_powerlaw(psi_tables["L"], psi_tables["psiT"], inverse=True)
            )
            channel["beta_over_nu_bar"] = dict(
                sweep=_sweep_powerlaw(psi_tables["L"], psi_tables["psiB"], inverse=True)
            )
        elif approach in ("effective_length", "mixed_subleading"):
            channel["nu_inv"] = dict(
                sweep=_sweep_powerlaw_subleading_fixed_omega(
                    deriv["L"], deriv["dlog"], bc_label, inverse=False,
                    use_effective_length=True, omega=GAP_RESIDUAL_OMEGA,
                )
            )
            channel["z_dynamic"] = dict(
                sweep=z_dynamic_subleading_sweep(
                    deriv["L"], deriv["gap"], bc_label, use_effective_length=True
                )
            )
            channel["beta_over_nu"] = dict(
                sweep=_sweep_powerlaw_subleading_fixed_omega(
                    psi_tables["L"], psi_tables["psiT"], bc_label, inverse=True,
                    use_effective_length=False, omega=raw_omega,
                )
            )
            channel["beta_over_nu_bar"] = dict(
                sweep=_sweep_powerlaw_subleading_fixed_omega(
                    psi_tables["L"], psi_tables["psiB"], bc_label, inverse=True,
                    use_effective_length=False, omega=raw_omega,
                )
            )
        else:
            channel["nu_inv"] = dict(
                sweep=_sweep_powerlaw_subleading_fixed_omega(
                    deriv["L"], deriv["dlog"], bc_label, inverse=False,
                    use_effective_length=False, omega=raw_omega,
                )
            )
            channel["z_dynamic"] = dict(
                sweep=z_dynamic_subleading_sweep(
                    deriv["L"], deriv["gap"], bc_label, use_effective_length=False
                )
            )
            channel["beta_over_nu"] = dict(
                sweep=_sweep_powerlaw_subleading_fixed_omega(
                    psi_tables["L"], psi_tables["psiT"], bc_label, inverse=True,
                    use_effective_length=False, omega=raw_omega,
                )
            )
            channel["beta_over_nu_bar"] = dict(
                sweep=_sweep_powerlaw_subleading_fixed_omega(
                    psi_tables["L"], psi_tables["psiB"], bc_label, inverse=True,
                    use_effective_length=False, omega=raw_omega,
                )
            )
        for block_name in ("nu_inv", "z_dynamic", "beta_over_nu", "beta_over_nu_bar"):
            _attach_approach_metadata(channel[block_name], channel)
        channels[approach] = channel

    nu_inv_final_key = "nu_inv_raw_subleading" if is_pbc else "nu_inv_mixed_subleading"
    z_dynamic_final_key = "z_dynamic_raw_subleading" if is_pbc else "z_dynamic_mixed_subleading"
    beta_over_nu_final_key = "beta_over_nu_raw_subleading" if is_pbc else "beta_over_nu_mixed_subleading"
    beta_over_nu_bar_final_key = "beta_over_nu_bar_raw_subleading" if is_pbc else "beta_over_nu_bar_mixed_subleading"

    primary = dict(
        bc=bc_label,
        channels=channels,
        g_crossings=raw_crossings,
        gc_sweep=gc_sweep_from_crossings(raw_crossings),
        chi_peak=chi_peaks,
        nu_inv_final_key=nu_inv_final_key,
        z_dynamic_final_key=z_dynamic_final_key,
        beta_over_nu_final_key=beta_over_nu_final_key,
        beta_over_nu_bar_final_key=beta_over_nu_bar_final_key,
    )

    mapping = {
        "nu_inv_leading": ("leading", "nu_inv"),
        "z_dynamic_leading": ("leading", "z_dynamic"),
        "beta_over_nu_leading": ("leading", "beta_over_nu"),
        "beta_over_nu_bar_leading": ("leading", "beta_over_nu_bar"),
        "nu_inv_raw_subleading": ("raw_subleading", "nu_inv"),
        "z_dynamic_raw_subleading": ("raw_subleading", "z_dynamic"),
        "beta_over_nu_raw_subleading": ("raw_subleading", "beta_over_nu"),
        "beta_over_nu_bar_raw_subleading": ("raw_subleading", "beta_over_nu_bar"),
    }
    if "raw_subleading_diagnostic" in channels:
        mapping.update({
            "nu_inv_raw_subleading_diagnostic": ("raw_subleading_diagnostic", "nu_inv"),
            "z_dynamic_raw_subleading_diagnostic": ("raw_subleading_diagnostic", "z_dynamic"),
            "beta_over_nu_raw_subleading_diagnostic": ("raw_subleading_diagnostic", "beta_over_nu"),
            "beta_over_nu_bar_raw_subleading_diagnostic": ("raw_subleading_diagnostic", "beta_over_nu_bar"),
        })
    if "mixed_subleading" in channels:
        mapping.update({
            "nu_inv_mixed_subleading": ("mixed_subleading", "nu_inv"),
            "z_dynamic_mixed_subleading": ("mixed_subleading", "z_dynamic"),
            "beta_over_nu_mixed_subleading": ("mixed_subleading", "beta_over_nu"),
            "beta_over_nu_bar_mixed_subleading": ("mixed_subleading", "beta_over_nu_bar"),
        })
    for key, (approach, block_name) in mapping.items():
        if approach in channels:
            primary[key] = channels[approach][block_name]

    if not is_pbc and "raw_subleading_diagnostic" in channels:
        primary["nu_inv_raw_subleading"] = channels["raw_subleading_diagnostic"]["nu_inv"]
        primary["z_dynamic_raw_subleading"] = channels["raw_subleading_diagnostic"]["z_dynamic"]
        primary["beta_over_nu_raw_subleading"] = channels["raw_subleading_diagnostic"]["beta_over_nu"]
        primary["beta_over_nu_bar_raw_subleading"] = channels["raw_subleading_diagnostic"]["beta_over_nu_bar"]

    primary["nu_inv_final"] = primary[nu_inv_final_key]
    primary["z_dynamic_final"] = primary[z_dynamic_final_key]
    primary["beta_over_nu_final"] = primary[beta_over_nu_final_key]
    primary["beta_over_nu_bar_final"] = primary[beta_over_nu_bar_final_key]
    primary["nu_inv"] = primary[nu_inv_final_key]
    primary["z_dynamic"] = primary[z_dynamic_final_key]
    primary["beta_over_nu"] = primary[beta_over_nu_final_key]
    primary["beta_over_nu_bar"] = primary[beta_over_nu_bar_final_key]
    final_approach = primary["nu_inv_final"].get("gpc_approach", "raw_subleading")
    primary["final_approach"] = final_approach
    primary["gap_derivative"] = channels.get(final_approach, next(iter(channels.values())))["gap_derivative"]
    primary["psi"] = channels.get(final_approach, next(iter(channels.values())))["psi"]

    final_lmin = _select_final_lmin(primary)
    primary["final_lmin"] = final_lmin

    for key in mapping:
        if key not in primary:
            continue
        sweep = primary[key]["sweep"]
        final = next((r for r in sweep if int(r["Lmin"]) == final_lmin), None)
        val = final["value"] if final else np.nan
        primary[key]["value"] = float(val) if np.isfinite(val) else np.nan
        primary[key]["lmin_drift"] = _lmin_drift(sweep, final_lmin)
    for alias in ("nu_inv_final", "z_dynamic_final", "beta_over_nu_final",
                  "beta_over_nu_bar_final", "nu_inv", "z_dynamic",
                  "beta_over_nu", "beta_over_nu_bar"):
        block = primary[alias]
        sweep = block["sweep"]
        final = next((r for r in sweep if int(r["Lmin"]) == final_lmin), None)
        val = final["value"] if final else np.nan
        block["value"] = float(val) if np.isfinite(val) else np.nan
        block["lmin_drift"] = _lmin_drift(sweep, final_lmin)

    for approach, channel in channels.items():
        channel["final_lmin"] = final_lmin
        for block_name in ("nu_inv", "z_dynamic", "beta_over_nu", "beta_over_nu_bar"):
            block = channel[block_name]
            row = next((r for r in block["sweep"] if int(r["Lmin"]) == final_lmin), None)
            block["value"] = float(row["value"]) if row and np.isfinite(row["value"]) else np.nan
            block["lmin_drift"] = _lmin_drift(block["sweep"], final_lmin)

    if not is_pbc and "mixed_subleading" in channels:
        mixed = channels["mixed_subleading"]
        if any(bool(r.get("use_effective_length", False)) for r in mixed["beta_over_nu"]["sweep"]):
            warnings.warn("OBC mixed_subleading beta_over_nu must use raw L, not effective length.", RuntimeWarning)
        if any(bool(r.get("use_effective_length", False)) for r in mixed["beta_over_nu_bar"]["sweep"]):
            warnings.warn("OBC mixed_subleading beta_over_nu_bar must use raw L, not effective length.", RuntimeWarning)
        if any(not bool(r.get("use_effective_length", False)) for r in mixed["nu_inv"]["sweep"]):
            warnings.warn("OBC mixed_subleading nu_inv must use effective length.", RuntimeWarning)
        if any(not bool(r.get("use_effective_length", False)) for r in mixed["z_dynamic"]["sweep"]):
            warnings.warn("OBC mixed_subleading z_dynamic must use effective length.", RuntimeWarning)

    primary["derived_exponents"] = build_derived_exponents_by_approach(primary)
    primary["gc_value"] = primary["nu_inv_final"].get("g_pc", np.nan)
    primary["gc_lmin_drift"] = np.nan
    return primary


def gamma_over_nu_hyperscaling(
    beta_over_nu_val: float,
    d: int = 1,
    z: float = Z_EXACT,
) -> dict:
    """Compute gamma/nu from hyperscaling: gamma/nu = (d+z) - 2*beta/nu."""
    if not np.isfinite(beta_over_nu_val) or not np.isfinite(z):
        return dict(value=np.nan, source="hyperscaling")
    gamma_over_nu = float(d + z) - 2.0 * beta_over_nu_val
    return dict(value=gamma_over_nu, source="hyperscaling")


def gamma_over_nu_sweep_from_hyperscaling(
    primary: dict,
    z_key: str = "z_dynamic_final",
    beta_key: str = "beta_over_nu_final",
) -> list[dict]:
    """Diagnostic Lmin sweep: gamma/nu(Lmin)=1+z(Lmin)-2 beta/nu(Lmin)."""
    if not primary:
        return []
    beta_rows = {
        int(r["Lmin"]): r
        for r in primary.get(beta_key, {}).get("sweep", [])
    }
    z_rows = {
        int(r["Lmin"]): r
        for r in primary.get(z_key, {}).get("sweep", [])
    }
    rows = []
    for Lmin in LMIN_SWEEP:
        beta_row = beta_rows.get(int(Lmin))
        z_row = z_rows.get(int(Lmin))
        if beta_row is None or z_row is None:
            continue
        beta_val = beta_row.get("value", np.nan)
        z_val = z_row.get("value", np.nan)
        if not (np.isfinite(beta_val) and np.isfinite(z_val)):
            continue
        rows.append(dict(
            Lmin=int(Lmin),
            value=float(1.0 + z_val - 2.0 * beta_val),
            dof=min(int(beta_row.get("dof", 0)), int(z_row.get("dof", 0))),
            n=min(int(beta_row.get("n", 0)), int(z_row.get("n", 0))),
        ))
    return rows


def _derived_exponent_row(Lmin: int, g_pc: float, nu_inv: float,
                          z: float, beta_over_nu: float) -> dict:
    """Compute approach-local exponents derived from primary FSS estimates."""
    if not all(np.isfinite(v) for v in (nu_inv, z, beta_over_nu)) or nu_inv == 0.0:
        nu = beta = gamma_over_nu = gamma = alpha_exp = delta_exp = np.nan
    else:
        nu = 1.0 / nu_inv
        gamma_over_nu = 1.0 + z - 2.0 * beta_over_nu
        beta = beta_over_nu * nu
        gamma = gamma_over_nu * nu
        alpha_exp = 2.0 - nu * (1.0 + z)
        delta_exp = 1.0 + gamma / beta if beta != 0.0 and np.isfinite(beta) else np.nan
    return dict(
        Lmin=int(Lmin),
        g_pc=float(g_pc) if np.isfinite(g_pc) else np.nan,
        nu_inv=float(nu_inv) if np.isfinite(nu_inv) else np.nan,
        z=float(z) if np.isfinite(z) else np.nan,
        beta_over_nu=float(beta_over_nu) if np.isfinite(beta_over_nu) else np.nan,
        gamma_over_nu=float(gamma_over_nu) if np.isfinite(gamma_over_nu) else np.nan,
        nu=float(nu) if np.isfinite(nu) else np.nan,
        beta=float(beta) if np.isfinite(beta) else np.nan,
        gamma=float(gamma) if np.isfinite(gamma) else np.nan,
        alpha=float(alpha_exp) if np.isfinite(alpha_exp) else np.nan,
        delta=float(delta_exp) if np.isfinite(delta_exp) else np.nan,
        value=float(alpha_exp) if np.isfinite(alpha_exp) else np.nan,
    )


def build_derived_exponents_by_approach(primary: dict) -> dict:
    """Build alpha and delta sweeps independently for each analysis approach."""
    if not primary:
        return {}

    final_lmin = int(primary.get("final_lmin", FINAL_LMIN_DEFAULT))
    derived: dict = {}
    for approach, channel in primary.get("channels", {}).items():
        nu_rows = {
            int(r["Lmin"]): r
            for r in channel.get("nu_inv", {}).get("sweep", [])
        }
        z_rows = {
            int(r["Lmin"]): r
            for r in channel.get("z_dynamic", {}).get("sweep", [])
        }
        beta_rows = {
            int(r["Lmin"]): r
            for r in channel.get("beta_over_nu", {}).get("sweep", [])
        }
        sweep = []
        for Lmin in LMIN_SWEEP:
            nu_row = nu_rows.get(int(Lmin))
            z_row = z_rows.get(int(Lmin))
            beta_row = beta_rows.get(int(Lmin))
            if nu_row is None or z_row is None or beta_row is None:
                continue
            row = _derived_exponent_row(
                int(Lmin),
                channel.get("g_pc", np.nan),
                nu_row.get("value", np.nan),
                z_row.get("value", np.nan),
                beta_row.get("value", np.nan),
            )
            if np.isfinite(row["alpha"]) or np.isfinite(row["delta"]):
                sweep.append(row)

        alpha_sweep = [dict(Lmin=r["Lmin"], value=r["alpha"]) for r in sweep]
        delta_sweep = [dict(Lmin=r["Lmin"], value=r["delta"]) for r in sweep]
        final = next((r for r in sweep if int(r["Lmin"]) == final_lmin), {})
        derived[approach] = dict(
            bc=primary.get("bc", ""),
            approach=approach,
            g_pc=float(channel.get("g_pc", np.nan)),
            gpc_source=str(channel.get("gpc_source", "")),
            sweep=sweep,
            final=final,
            alpha=dict(
                value=float(final.get("alpha", np.nan)) if final else np.nan,
                lmin_drift=_lmin_drift(alpha_sweep, final_lmin),
                sweep=alpha_sweep,
            ),
            delta=dict(
                value=float(final.get("delta", np.nan)) if final else np.nan,
                lmin_drift=_lmin_drift(delta_sweep, final_lmin),
                sweep=delta_sweep,
            ),
        )
    return derived


def write_primary_tables(primary: dict) -> None:
    bc = primary["bc"].lower()
    path = fss_path_for_bc(primary["bc"], f"primary_tables_{bc}.dat")
    chi = primary["chi_peak"]
    psi = primary["psi"]
    deriv = primary["gap_derivative"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("# approach_gpc: approach  g_pc  source  crossing_scale\n")
        for approach, channel in primary.get("channels", {}).items():
            out_approach = public_approach_label(primary["bc"], approach)
            f.write(
                f"gpc_approach {out_approach:<28} {channel.get('g_pc', np.nan):.12f} "
                f"{channel.get('gpc_source', ''):<18} {channel.get('gpc_crossing_scale', '')}\n"
            )
        f.write("# chi_peak_table: chi_z FD is plot-only\n")
        for L, gp, ch in zip(chi["L"], chi["g_peak"], chi["chi_max"]):
            f.write(f"chi_peak  {int(L):4d}  {gp:.12f}  {ch:.12e}\n")
        f.write("#\n# psi_gc_table: L  psi_tilde(gc)  psi_bar(gc)\n")
        for L, psi_t, psi_b in zip(psi["L"], psi["psiT"], psi["psiB"]):
            f.write(f"psi_gc    {int(L):4d}  {psi_t:.12e}  {psi_b:.12e}\n")
        f.write("#\n# gap_derivative_table: L  Delta0(gc)  abs_dlogDelta_dg\n")
        for L, gap, dlog in zip(deriv["L"], deriv["gap"], deriv["dlog"]):
            f.write(f"gap_deriv {int(L):4d}  {gap:.12e}  {dlog:.12e}\n")
        block_labels = {
            "nu_inv": "nu_inv",
            "z_dynamic": "z_dynamic",
            "beta_over_nu": "beta_over_nu_psi_tilde",
            "beta_over_nu_bar": "beta_over_nu_psi_bar",
        }
        for approach, approach_data in primary.get("channels", {}).items():
            out_approach = public_approach_label(primary["bc"], approach)
            for block_name, base_label in block_labels.items():
                block = approach_data.get(block_name, {})
                if not block:
                    continue
                drift = block.get("lmin_drift", np.nan)
                g_pc = block.get("g_pc", np.nan)
                source = block.get("gpc_source", "")
                label = f"{base_label}_{out_approach}"
                f.write(f"#\n# sweep_{label}: Lmin  value  lmin_drift_final  g_pc  gpc_source  resid_rms  n\n")
                for row in block.get("sweep", []):
                    f.write(
                        f"sweep_{label} {row['Lmin']:4d}  {row['value']:.12e}  "
                        f"{drift:.12e}  {g_pc:.12f}  {source:<18}  "
                        f"{row.get('resid_rms', np.nan):.12e}  "
                        f"{row.get('n', row.get('npts', 0)):3d}\n"
                    )
    print(f"  [OK] {path.relative_to(FSS_DIR)}")


def write_nu_inv_sweep(primary_pbc: dict, primary_obc: dict) -> None:
    path = FSS_DIR / "nu_inv_gap_derivative_sweep.dat"
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Gap-derivative 1/nu sweeps from |d_g log Delta0| at gc\n")
        f.write("# Columns: BC approach Lmin g_pc gpc_source value B omega Le_shift resid_rms n is_final_approach is_final_lmin\n")
        for primary in (primary_pbc, primary_obc):
            if not primary:
                continue
            bc = primary["bc"]
            final_key = primary.get("nu_inv_final_key", "")
            final_lmin = int(primary.get("final_lmin", FINAL_LMIN_DEFAULT))
            for approach, data in primary.get("channels", {}).items():
                out_approach = public_approach_label(bc, approach)
                block = data.get("nu_inv", {})
                is_final_approach = int(approach_block_key(approach, "nu_inv") == final_key)
                for row in block.get("sweep", []):
                    is_final_lmin = int(int(row.get("Lmin", 0)) == final_lmin)
                    f.write(
                        f"{bc:<3} {out_approach:<28} {int(row.get('Lmin', 0)):4d} "
                        f"{block.get('g_pc', np.nan):.12f} "
                        f"{block.get('gpc_source', ''):<18} "
                        f"{row.get('value', np.nan):.12e} "
                        f"{row.get('B', np.nan):.12e} "
                        f"{row.get('omega_fixed', np.nan):.6f} "
                        f"{row.get('Le_shift_used', np.nan):.6f} "
                        f"{row.get('resid_rms', np.nan):.12e} "
                        f"{int(row.get('npts', row.get('n', 0)))} "
                        f"{is_final_approach} {is_final_lmin}\n"
                    )
    print(f"  [OK] {path.name}")


def write_beta_over_nu_sweep(primary_pbc: dict, primary_obc: dict) -> None:
    path = FSS_DIR / "beta_over_nu_sweep.dat"
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Order-parameter beta/nu sweeps at gc using raw L only\n")
        f.write("# psi_tilde is the primary observable; psi_bar is a cross-check.\n")
        f.write("# Columns: BC observable role approach Lmin g_pc gpc_source value B omega Le_shift resid_rms n is_final_approach is_final_lmin\n")
        for primary in (primary_pbc, primary_obc):
            if not primary:
                continue
            bc = primary["bc"]
            final_tilde = primary.get("beta_over_nu_final_key", "")
            final_bar = primary.get("beta_over_nu_bar_final_key", "")
            final_lmin = int(primary.get("final_lmin", FINAL_LMIN_DEFAULT))
            for approach, data in primary.get("channels", {}).items():
                out_approach = public_approach_label(bc, approach)
                for observable, role, block_name, final_key in [
                    ("psi_tilde", "primary", "beta_over_nu", final_tilde),
                    ("psi_bar", "cross_check", "beta_over_nu_bar", final_bar),
                ]:
                    block = data.get(block_name, {})
                    is_final_approach = int(approach_block_key(approach, block_name) == final_key)
                    for row in block.get("sweep", []):
                        is_final_lmin = int(int(row.get("Lmin", 0)) == final_lmin)
                        f.write(
                            f"{bc:<3} {observable:<9} {role:<11} {out_approach:<28} "
                            f"{int(row.get('Lmin', 0)):4d} "
                            f"{block.get('g_pc', np.nan):.12f} "
                            f"{block.get('gpc_source', ''):<18} "
                            f"{row.get('value', np.nan):.12e} "
                            f"{row.get('B', np.nan):.12e} "
                            f"{row.get('omega_fixed', np.nan):.6f} "
                            f"{row.get('Le_shift_used', np.nan):.6f} "
                            f"{row.get('resid_rms', np.nan):.12e} "
                            f"{int(row.get('npts', row.get('n', 0)))} "
                            f"{is_final_approach} {is_final_lmin}\n"
                        )
    print(f"  [OK] {path.name}")


def write_exponent_sweeps(primary_pbc: dict, primary_obc: dict) -> None:
    path = FSS_DIR / "exponent_sweeps.dat"

    quantity_labels = {
        "nu_inv": "nu_inv",
        "z_dynamic": "z_dynamic",
        "beta_over_nu": "beta_over_nu_psi_tilde",
        "beta_over_nu_bar": "beta_over_nu_psi_bar",
    }

    def _write_block_rows(f, primary: dict, quantity: str, approach: str, block: dict) -> None:
        drift = block.get("lmin_drift", np.nan)
        for row in block.get("sweep", []):
            f.write(
                f"{primary['bc']:<3} {quantity:<24} {approach:<18} "
                f"{int(row.get('Lmin', 0)):4d} "
                f"{block.get('g_pc', np.nan):.12f} "
                f"{block.get('gpc_source', ''):<18} "
                f"{row.get('value', np.nan):.12e} "
                f"{drift:.12e} "
                f"{row.get('omega_fixed', np.nan):.12e} "
                f"{row.get('Le_shift_used', 0.0 if approach != 'effective_length' else np.nan):.12e} "
                f"{row.get('B', np.nan):.12e} "
                f"{row.get('resid_rms', np.nan):.12e} "
                f"{int(row.get('npts', row.get('n', 0)))}\n"
            )

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Consolidated deterministic exponent sweeps\n")
        f.write("# Columns: BC quantity approach Lmin g_pc gpc_source value lmin_drift_final omega Le_shift B resid_rms n\n")
        for primary in (primary_pbc, primary_obc):
            if not primary:
                continue
            bc_label = primary["bc"]
            for approach, approach_data in primary.get("channels", {}).items():
                out_approach = public_approach_label(bc_label, approach)
                for block_name, quantity in quantity_labels.items():
                    block = approach_data.get(block_name, {})
                    if block:
                        _write_block_rows(f, primary, quantity, out_approach, block)

                z_key = approach_block_key(approach, "z_dynamic")
                beta_key = approach_block_key(approach, "beta_over_nu")
                if z_key not in primary or beta_key not in primary:
                    continue
                sweep = gamma_over_nu_sweep_from_hyperscaling(primary, z_key, beta_key)
                drift = _lmin_drift(sweep, primary.get("final_lmin", FINAL_LMIN_DEFAULT))
                z_block = primary.get(z_key, {})
                for row in sweep:
                    f.write(
                        f"{primary['bc']:<3} {'gamma_over_nu':<24} {out_approach:<18} "
                        f"{int(row.get('Lmin', 0)):4d} "
                        f"{z_block.get('g_pc', np.nan):.12f} "
                        f"{z_block.get('gpc_source', ''):<18} "
                        f"{row.get('value', np.nan):.12e} "
                        f"{drift:.12e} "
                        f"{np.nan:.12e} {np.nan:.12e} {np.nan:.12e} {np.nan:.12e} "
                        f"{int(row.get('n', 0))}\n"
                    )
    print(f"  [OK] {path.name}")


def write_derived_exponents_sweep(primary_pbc: dict, primary_obc: dict) -> None:
    path = FSS_DIR / "derived_exponents_sweep.dat"
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Approach-local derived exponent sweeps\n")
        f.write(
            "# BC quantity approach Lmin g_pc value lmin_drift_final "
            "nu_inv z beta_over_nu gamma_over_nu nu beta gamma alpha delta\n"
        )
        for primary in (primary_pbc, primary_obc):
            if not primary:
                continue
            bc = primary.get("bc", "")
            for approach, derived in primary.get("derived_exponents", {}).items():
                out_approach = public_approach_label(bc, approach)
                alpha_drift = derived.get("alpha", {}).get("lmin_drift", np.nan)
                delta_drift = derived.get("delta", {}).get("lmin_drift", np.nan)
                for row in derived.get("sweep", []):
                    for quantity, drift in (("alpha", alpha_drift), ("delta", delta_drift)):
                        f.write(
                            f"{bc:<3} {quantity:<8} {out_approach:<28} "
                            f"{int(row.get('Lmin', 0)):4d} "
                            f"{row.get('g_pc', np.nan):.12f} "
                            f"{row.get(quantity, np.nan):.12e} "
                            f"{drift:.12e} "
                            f"{row.get('nu_inv', np.nan):.12e} "
                            f"{row.get('z', np.nan):.12e} "
                            f"{row.get('beta_over_nu', np.nan):.12e} "
                            f"{row.get('gamma_over_nu', np.nan):.12e} "
                            f"{row.get('nu', np.nan):.12e} "
                            f"{row.get('beta', np.nan):.12e} "
                            f"{row.get('gamma', np.nan):.12e} "
                            f"{row.get('alpha', np.nan):.12e} "
                            f"{row.get('delta', np.nan):.12e}\n"
                        )
    print(f"  [OK] {path.name}")


def write_z_dynamic_sweep(primary_pbc: dict, primary_obc: dict) -> None:
    path = FSS_DIR / "z_dynamic_sweep.dat"
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Dynamic exponent z diagnostic sweeps from Delta0(gc,L)\n")
        f.write("# Columns: BC approach Lmin g_pc gpc_source value B omega Le_shift resid_rms n is_final_approach is_final_lmin\n")
        for primary in (primary_pbc, primary_obc):
            if not primary:
                continue
            bc = primary["bc"]
            final_key = primary.get("z_dynamic_final_key", "")
            final_lmin = int(primary.get("final_lmin", FINAL_LMIN_DEFAULT))
            for approach, data in primary.get("channels", {}).items():
                out_approach = public_approach_label(bc, approach)
                block = data.get("z_dynamic", {})
                is_final_approach = int(approach_block_key(approach, "z_dynamic") == final_key)
                for row in block.get("sweep", []):
                    is_final_lmin = int(int(row.get("Lmin", 0)) == final_lmin)
                    f.write(
                        f"{bc:<3} {out_approach:<28} {int(row.get('Lmin', 0)):4d} "
                        f"{block.get('g_pc', np.nan):.12f} "
                        f"{block.get('gpc_source', ''):<18} "
                        f"{row.get('value', row.get('z', np.nan)):.12e} "
                        f"{row.get('B', np.nan):.12e} "
                        f"{row.get('omega_fixed', np.nan):.6f} "
                        f"{row.get('Le_shift_used', np.nan):.6f} "
                        f"{row.get('resid_rms', np.nan):.12e} "
                        f"{int(row.get('npts', row.get('n', 0)))} "
                        f"{is_final_approach} {is_final_lmin}\n"
                    )
    print(f"  [OK] {path.name}")


def _main_chiperp_approaches(primary: dict) -> tuple[str, ...]:
    """Approaches for the public chi_perp logarithmic alpha diagnostic."""
    if not primary:
        return tuple()
    bc_key = str(primary.get("bc", "")).upper()
    if bc_key == "PBC":
        candidates = ("leading", "raw_subleading")
    elif bc_key == "OBC":
        candidates = ("leading", "mixed_subleading")
    else:
        candidates = tuple(primary.get("channels", {}).keys())
    return tuple(a for a in candidates if a in primary.get("channels", {}))


def _chiperp_log_row_for_lmin(log_block: dict, fit_kind: str, Lmin: int) -> dict:
    rows = log_block.get(fit_kind, []) if log_block else []
    row = next((r for r in rows if int(r.get("Lmin", -1)) == int(Lmin)), None)
    if row is None:
        row = next((r for r in rows if int(r.get("Lmin", -1)) >= int(Lmin)), None)
    if row is None and rows:
        row = rows[-1]
    return row or {}


def attach_chiperp_log_diagnostics(primary: dict, obs_data: dict) -> None:
    """Attach raw-L logarithmic C_perp diagnostics to the main channels."""
    if not primary:
        return
    final_lmin = int(primary.get("final_lmin", FINAL_LMIN_DEFAULT))
    for approach in _main_chiperp_approaches(primary):
        channel = primary.get("channels", {}).get(approach, {})
        g_pc = float(channel.get("g_pc", np.nan))
        table = chiperp_critical_table(obs_data, g_pc, use_g_chi=True)
        log_block = chiperp_log_sweep(table["L"], table["C_perp"], primary.get("bc", ""))
        final_leading = _chiperp_log_row_for_lmin(log_block, "leading", final_lmin)
        final_subleading = _chiperp_log_row_for_lmin(log_block, "subleading", final_lmin)
        log_block.update(
            table=table,
            g_pc=g_pc,
            use_g_chi=True,
            final_lmin=final_lmin,
            final_leading=final_leading,
            final_subleading=final_subleading,
            alpha_chiperp_log=0.0,
            value=0.0,
            sweep=[*log_block["leading"], *log_block["subleading"]],
        )
        channel["chiperp_log"] = log_block


def write_alpha_chiperp_sweep(primary_pbc: dict, primary_obc: dict) -> None:
    path = FSS_DIR / "alpha_chiperp_sweep.dat"
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Transverse susceptibility logarithmic diagnostic for alpha\n")
        f.write("# y(L)=C_x(L)=g_pc*chi_x(L,g_pc)\n")
        f.write("# alpha_log=0 because the fitted critical singularity is logarithmic, not power-law\n")
        f.write("# Columns:\n")
        f.write("# BC approach fit_type Lmin g_pc n a0 a1 b omega resid_rms alpha_log role\n")
        for primary in (primary_pbc, primary_obc):
            if not primary:
                continue
            bc = primary.get("bc", "")
            for approach in _main_chiperp_approaches(primary):
                out_approach = public_approach_label(bc, approach)
                log_block = primary.get("channels", {}).get(approach, {}).get("chiperp_log", {})
                for fit_kind in ("leading", "subleading"):
                    for row in log_block.get(fit_kind, []):
                        f.write(
                            f"{bc:<3} {out_approach:<28} {row.get('fit_type', ''):<14} "
                            f"{int(row.get('Lmin', 0)):4d} "
                            f"{log_block.get('g_pc', np.nan):.12f} "
                            f"{int(row.get('n', 0)):3d} "
                            f"{row.get('a0', np.nan):.12e} "
                            f"{row.get('a1', np.nan):.12e} "
                            f"{row.get('b', np.nan):.12e} "
                            f"{row.get('omega', np.nan):.6f} "
                            f"{row.get('resid_rms', np.nan):.12e} "
                            f"{row.get('alpha_log', np.nan):.12e} "
                            f"{row.get('role', 'logarithmic_alpha_zero')}\n"
                        )
    print(f"  [OK] {path.name}")


def _eval_chiperp_log_fit_row(L, row: dict) -> np.ndarray:
    """Evaluate one chi_perp logarithmic diagnostic fit row on raw L."""
    L_arr = np.asarray(L, dtype=float)
    if not row or not (
        np.isfinite(row.get("a0", np.nan))
        and np.isfinite(row.get("a1", np.nan))
    ):
        return np.full_like(L_arr, np.nan, dtype=float)
    y = row["a0"] + row["a1"] * np.log(L_arr)
    b = row.get("b", np.nan)
    omega = row.get("omega", np.nan)
    if np.isfinite(b) and np.isfinite(omega):
        y = y + b * L_arr ** (-omega)
    return y


def plot_alpha_chiperp_logfit(primary_pbc: dict, primary_obc: dict) -> None:
    """Plot C_x(L) data with separate leading/subleading logarithmic fits."""
    rc_params = {
        "font.size": 8.5,
        "axes.labelsize": 9.0,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 6.8,
    }
    axis_label_fontsize = 9.0
    tick_fontsize = 8.0
    title_fontsize = 9.0
    axis_linewidth = float(plt.rcParams.get("axes.linewidth", 0.8))
    major_tick_width = float(plt.rcParams.get("xtick.major.width", axis_linewidth))
    minor_tick_width = float(plt.rcParams.get("xtick.minor.width", major_tick_width))
    major_tick_length = float(plt.rcParams.get("xtick.major.size", 3.5))
    minor_tick_length = float(plt.rcParams.get("xtick.minor.size", 2.0))

    def _fit_label(fit_kind: str) -> str:
        if fit_kind == "leading":
            return r"fit: $a_0+a_1\log L$"
        return r"fit: $a_0+a_1\log L+bL^{-\omega}$"

    panel_specs = [
        (primary_pbc, "PBC", "leading", "leading", plt.cm.plasma(0.56)),
        (primary_pbc, "PBC", "raw_subleading", "subleading", plt.cm.plasma(0.74)),
        (primary_obc, "OBC", "leading", "leading", plt.cm.viridis(0.56)),
        (primary_obc, "OBC", "mixed_subleading", "subleading", plt.cm.viridis(0.30)),
    ]
    cutoff_color = "#9a9a9a"

    with plt.rc_context(rc_params):
        fig, axes = plt.subplots(
            1, 4, figsize=(7.4, 2.25), constrained_layout=True, sharey=True
        )
        fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.03, wspace=0.03, hspace=0.02)

        for idx, (ax, (primary, bc_label, approach, fit_kind, color)) in enumerate(zip(axes, panel_specs)):
            if not primary or approach not in primary.get("channels", {}):
                ax.text(0.5, 0.5, "No finite data", transform=ax.transAxes,
                        ha="center", va="center", color="gray")
                ax.set_axis_off()
                continue

            log_block = primary.get("channels", {}).get(approach, {}).get("chiperp_log", {})
            row_key = "final_leading" if fit_kind == "leading" else "final_subleading"
            fit_row = log_block.get(row_key, {})
            line_style = "--" if fit_kind == "leading" else "-"

            table = log_block.get("table", {})
            L_arr = np.asarray(table.get("L", []), dtype=float)
            y_arr = np.asarray(table.get("C_perp", []), dtype=float)
            final_lmin = int(log_block.get("final_lmin", FINAL_LMIN_DEFAULT))
            finite = np.isfinite(L_arr) & np.isfinite(y_arr) & (L_arr > 0.0)

            low = finite & (L_arr < float(final_lmin))
            high = finite & (L_arr >= float(final_lmin))
            if np.count_nonzero(low):
                ax.plot(L_arr[low], y_arr[low], ls="None", marker="o",
                        markerfacecolor="none", markeredgecolor=color,
                        markeredgewidth=1.1, alpha=0.50, label="_nolegend_")
            if np.count_nonzero(high):
                ax.plot(L_arr[high], y_arr[high], ls="None", marker="o",
                        color=color, markersize=3.4, label="_nolegend_")

            if np.count_nonzero(high) >= 2:
                L_fit = np.geomspace(float(np.min(L_arr[high])),
                                     float(np.max(L_arr[high])), 300)
                y_fit = _eval_chiperp_log_fit_row(L_fit, fit_row)
                m_fit = np.isfinite(y_fit)
                if np.count_nonzero(m_fit):
                    ax.plot(
                        L_fit[m_fit],
                        y_fit[m_fit],
                        color=color,
                        lw=1.35,
                        ls=line_style,
                        label=_fit_label(fit_kind),
                    )

            ax.axvline(final_lmin, color=cutoff_color, ls=":", lw=1.2, alpha=0.75)
            ax.set_xscale("log")
            ax.set_xticks([4, 6, 8, 10, 20])
            ax.set_xticklabels(["4", "6", "8", "10", "20"])
            ax.set_xlabel(r"$L$", fontsize=axis_label_fontsize)
            ax.set_ylabel(r"$C_x(L)$" if idx == 0 else "", fontsize=axis_label_fontsize)
            if bc_label == "OBC" and approach == "mixed_subleading":
                out_approach = "subleading"
            else:
                out_approach = public_approach_label(bc_label, approach).replace("_", " ")
            ax.set_title(f"{bc_label} {out_approach}", loc="left",
                         fontsize=title_fontsize)
            for spine in ax.spines.values():
                spine.set_linewidth(axis_linewidth)
            ax.tick_params(
                direction="in",
                which="major",
                top=True,
                right=True,
                labelsize=tick_fontsize,
                width=major_tick_width,
                length=major_tick_length,
            )
            ax.tick_params(
                direction="in",
                which="minor",
                top=True,
                right=True,
                width=minor_tick_width,
                length=minor_tick_length,
            )
            apply_grid(ax)
            legend_kwargs = dict(
                frameon=False,
                loc="best",
                handlelength=2.2,
                borderaxespad=0.2,
            )
            if bc_label == "PBC":
                legend_kwargs.update(loc="lower left", bbox_to_anchor=(0.0, 0.06))
            elif bc_label == "OBC":
                legend_kwargs.update(loc="upper left", bbox_to_anchor=(0.0, 0.90))
            ax.legend(**legend_kwargs)

    save_fig(fig, "alpha_chiperp_logfit.pdf")


def write_gap_crossing_table(rows: list[dict], bc_label: str) -> None:
    path = fss_path_for_bc(bc_label, f"gap_crossings_{bc_label.lower()}.dat")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Deterministic crossings of X*Delta0 spline curves\n")
        f.write("# Columns: L1  L2  L_eff  scale_eff  use_effective_length  g_pc\n")
        for r in rows:
            f.write(
                f"{r['L1']:4d}  {r['L2']:4d}  {r['L_eff']:.6f}  "
                f"{r.get('scale_eff', r['L_eff']):.6f}  "
                f"{int(bool(r.get('use_effective_length', False)))}  "
                f"{r['g_pc']:.12f}\n"
            )
    print(f"  [OK] {path.relative_to(FSS_DIR)}")


def _plot_sweep_axis(ax, sweep: list[dict], *, color, label,
                     final_lmin=FINAL_LMIN_DEFAULT, marker="o",
                     alpha=1.0):
    _ = final_lmin
    x = np.asarray([r["Lmin"] for r in sweep], dtype=float)
    y = np.asarray([r["value"] for r in sweep], dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(m) == 0:
        return
    ax.plot(x[m], y[m], marker=None, ls="-", lw=1.15, color=color,
            alpha=0.48, zorder=1)
    ax.plot(x[m], y[m], marker=marker, ls="None", color=color,
            markersize=5.4, markeredgewidth=0.9,
            label=label, alpha=alpha, zorder=2)


def plot_primary_robustness(primary_pbc: dict, primary_obc: dict) -> None:
    """Convergence plots for 1/nu, beta/nu, z, and hyperscaling gamma/nu vs Lmin."""
    from matplotlib.lines import Line2D

    specs = [
        (primary_pbc, plt.cm.plasma, "PBC"),
        (primary_obc, plt.cm.viridis, "OBC"),
    ]
    color_pos = dict(
        leading=0.55,
        raw=0.78,
        effective=0.28,
        psit=0.34,
        psit_leading=0.56,
        psib_leading=0.70,
        psib=0.88,
        gamma_leading=0.55,
        gamma_raw=0.78,
        gamma_effective=0.28,
    )
    legend_fontsize = EXPONENT_CONVERGENCE_LEGEND_FONTSIZE
    out_dir = "exponent_convergence"
    cutoff_color = "#8a8a8a"

    def _plot_block(
        ax,
        primary: dict,
        cmap,
        bc_label: str,
        block_key: str,
        label: str,
        *,
        color_key: str,
        marker: str = "o",
        alpha: float = 1.0,
    ) -> None:
        if block_key not in primary:
            return
        block = primary.get(block_key, {})
        final_lmin = primary.get("final_lmin", FINAL_LMIN_DEFAULT)
        _ = bc_label
        _plot_sweep_axis(
            ax,
            block.get("sweep", []),
            color=cmap(color_pos[color_key]),
            label=label,
            final_lmin=final_lmin,
            marker=marker,
            alpha=alpha,
        )

    def _new_axis():
        fig, ax = plt.subplots(figsize=(4.8, 3.8), constrained_layout=True)
        return fig, ax

    def _style_legend(legend) -> None:
        legend.get_frame().set_linewidth(0.0)

    def _curve_handle(cmap, color_key: str, marker: str, label: str):
        return Line2D([], [], color=cmap(color_pos[color_key]), marker=marker,
                      ls="None", lw=0, markersize=5.4, label=label)

    def _reference_handle(label: str):
        return Line2D([], [], color="gray", ls="--", lw=1.4, label=label)

    def _cutoff_handle():
        return Line2D([], [], color=cutoff_color, ls=":", lw=1.4,
                      label=r"$L^{\star}$")

    def _add_cutoff_line(ax) -> None:
        ax.axvline(FINAL_LMIN_DEFAULT, color=cutoff_color, ls=":",
                   lw=1.25, alpha=0.82, zorder=0.35)

    def _add_external_convergence_legend(
        fig: plt.Figure,
        pbc_handles: list,
        obc_handles: list,
        reference_handles: list,
    ) -> None:
        x_anchors = [
            EXPONENT_CONVERGENCE_LEGEND_X_CENTER - EXPONENT_CONVERGENCE_LEGEND_X_SPACING,
            EXPONENT_CONVERGENCE_LEGEND_X_CENTER,
            EXPONENT_CONVERGENCE_LEGEND_X_CENTER + EXPONENT_CONVERGENCE_LEGEND_X_SPACING,
        ]
        legend_specs = [
            (pbc_handles, "PBC", x_anchors[0], True),
            (obc_handles, "OBC", x_anchors[1], True),
            (reference_handles, "", x_anchors[2], False),
        ]
        legends = []
        for handles, title, x_anchor, add_label in legend_specs:
            handles = [h for h in handles if h is not None]
            if not handles:
                continue
            leg = fig.legend(
                handles=handles,
                frameon=False,
                fontsize=legend_fontsize,
                loc="center",
                bbox_to_anchor=(x_anchor, EXPONENT_CONVERGENCE_LEGEND_Y_ANCHOR),
                borderaxespad=0.0,
                handlelength=1.45,
                handletextpad=0.45,
                labelspacing=0.45,
                ncol=1,
            )
            _style_legend(leg)
            legends.append((leg, title, add_label))

        if not legends:
            return

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for leg, title, add_label in legends:
            if not add_label or not title:
                continue
            bbox = leg.get_window_extent(renderer=renderer)
            bbox_fig = bbox.transformed(fig.transFigure.inverted())
            x_text = bbox_fig.x0 - EXPONENT_CONVERGENCE_LEGEND_LABEL_XPAD
            y_text = 0.5 * (bbox_fig.y0 + bbox_fig.y1)
            fig.text(
                x_text,
                y_text,
                title,
                rotation=90,
                va="center",
                ha="right",
                fontsize=legend_fontsize,
            )

    def _numeric_band(ax, value: float, drift: float, *, color, label: str) -> str | None:
        if not np.isfinite(value):
            return None
        if np.isfinite(drift) and drift > 0.0:
            ax.axhspan(value - drift, value + drift,
                       color=color, alpha=0.12, lw=0, zorder=0.15)
        ax.axhline(value, color=color, ls="-", lw=1.15,
                   alpha=0.72, zorder=0.25)
        return Line2D([], [], color=color, lw=5, alpha=0.25, label=label)

    def _final_estimate(primary: dict, block_key: str) -> tuple[float, float]:
        if not primary:
            return np.nan, np.nan
        block = primary.get(block_key, {})
        row = _row_for_lmin(block, primary.get("final_lmin", FINAL_LMIN_DEFAULT))
        if row is not None:
            return float(row.get("value", np.nan)), float(block.get("lmin_drift", np.nan))
        return float(block.get("value", np.nan)), float(block.get("lmin_drift", np.nan))

    def _gamma_final_estimate(primary: dict) -> tuple[float, float]:
        if not primary:
            return np.nan, np.nan
        sweep = gamma_over_nu_sweep_from_hyperscaling(primary)
        final_lmin = primary.get("final_lmin", FINAL_LMIN_DEFAULT)
        row = next((r for r in sweep if int(r["Lmin"]) == int(final_lmin)), None)
        if row is None and sweep:
            row = sweep[-1]
        value = float(row.get("value", np.nan)) if row else np.nan
        return value, _lmin_drift(sweep, final_lmin)

    def _finish_axis(fig, ax, ylabel: str, filename: str, *, legend_loc: str = "best",
                     ytop: float | None = None, build_legend: bool = True) -> None:
        ax.set_xlabel(r"$L_{\min}$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(LMIN_SWEEP)
        if ytop is not None:
            ax.set_ylim(top=ytop)
        ax.tick_params(direction="in", which="both", top=True, right=True,
                       labelsize=TICK_FONTSIZE)
        apply_grid(ax)
        if build_legend:
            leg = ax.legend(frameon=True, fontsize=legend_fontsize, loc=legend_loc,
                            facecolor="white", edgecolor="none", framealpha=0.74)
            leg.get_frame().set_linewidth(0.0)
        save_fig(fig, f"{out_dir}/{filename}")

    fig, ax = _new_axis()
    for primary, cmap, bc_label in specs:
        if not primary:
            continue
        _plot_block(ax, primary, cmap, bc_label, "nu_inv_leading",
                    "_nolegend_", color_key="leading",
                    marker="s", alpha=0.78)
        if bc_label == "OBC":
            _plot_block(ax, primary, cmap, bc_label, "nu_inv_mixed_subleading",
                        "_nolegend_",
                        color_key="effective", marker="D")
        else:
            _plot_block(ax, primary, cmap, bc_label, "nu_inv_raw_subleading",
                        "_nolegend_", color_key="raw", marker="o")
    ax.axhline(1.0 / NU, color="gray", ls="--", lw=1.2, alpha=0.8,
               label="_nolegend_")
    pbc_val, pbc_drift = _final_estimate(primary_pbc, "nu_inv_final")
    obc_val, obc_drift = _final_estimate(primary_obc, "nu_inv_final")
    band_handles = [
        _numeric_band(ax, pbc_val, pbc_drift, color=plt.cm.plasma(0.83),
                      label="PBC Final"),
        _numeric_band(ax, obc_val, obc_drift, color=plt.cm.viridis(0.52),
                      label="OBC Final"),
    ]
    _add_cutoff_line(ax)
    _add_external_convergence_legend(
        fig,
        [
            _curve_handle(plt.cm.plasma, "leading", "s", "leading"),
            _curve_handle(plt.cm.plasma, "raw", "o",
                          rf"subl. $\omega={OMEGA_PBC:.0f}$"),
        ],
        [
            _curve_handle(plt.cm.viridis, "leading", "s", "leading"),
            _curve_handle(plt.cm.viridis, "effective", "D", "mixed subl."),
        ],
        [
            _reference_handle(rf"theory $={1.0 / NU:.4f}$"),
            _cutoff_handle(),
            *band_handles,
        ],
    )
    _finish_axis(fig, ax, r"$1/\nu$", "nu_inv_vs_Lmin.pdf",
                 ytop=1.02, build_legend=False)

    fig, ax = _new_axis()
    for primary, cmap, bc_label in specs:
        if not primary:
            continue
        _plot_block(ax, primary, cmap, bc_label, "beta_over_nu_leading",
                    "_nolegend_",
                    color_key="psit_leading", marker="s", alpha=0.76)
        if bc_label == "OBC":
            _plot_block(ax, primary, cmap, bc_label, "beta_over_nu_mixed_subleading",
                        "_nolegend_",
                        color_key="psit", marker="o")
        else:
            _plot_block(ax, primary, cmap, bc_label, "beta_over_nu_raw_subleading",
                        "_nolegend_",
                        color_key="psit", marker="o")
        _plot_block(ax, primary, cmap, bc_label, "beta_over_nu_bar_leading",
                    "_nolegend_",
                    color_key="psib_leading", marker="v", alpha=0.78)
        if bc_label == "OBC":
            _plot_block(ax, primary, cmap, bc_label, "beta_over_nu_bar_mixed_subleading",
                        "_nolegend_",
                        color_key="psib", marker="^", alpha=0.82)
        else:
            _plot_block(ax, primary, cmap, bc_label, "beta_over_nu_bar_raw_subleading",
                        "_nolegend_",
                        color_key="psib", marker="^", alpha=0.82)
    ax.axhline(BETA / NU, color="gray", ls="--", lw=1.2, alpha=0.8,
               label="_nolegend_")
    pbc_val, pbc_drift = _final_estimate(primary_pbc, "beta_over_nu_final")
    obc_val, obc_drift = _final_estimate(primary_obc, "beta_over_nu_final")
    band_handles = [
        _numeric_band(ax, pbc_val, pbc_drift, color=plt.cm.plasma(0.83),
                      label="PBC Final"),
        _numeric_band(ax, obc_val, obc_drift, color=plt.cm.viridis(0.52),
                      label="OBC Final"),
    ]

    def _beta_handles_for(cmap, bc_label: str, omega: float) -> list:
        sub_label = "mixed subl." if bc_label == "OBC" else rf"subl. $\omega={omega:.0f}$"
        return [
            Line2D([], [], color=cmap(color_pos["psit_leading"]), marker="s",
                   ls="None", lw=0, markersize=5.4,
                   label=r"$\tilde{\Psi}$ lead."),
            Line2D([], [], color=cmap(color_pos["psit"]), marker="o",
                   ls="None", lw=0, markersize=5.4,
                   label=rf"$\tilde{{\Psi}}$ {sub_label}"),
            Line2D([], [], color=cmap(color_pos["psib_leading"]), marker="v",
                   ls="None", lw=0, markersize=5.4,
                   label=r"$\bar{\Psi}$ lead."),
            Line2D([], [], color=cmap(color_pos["psib"]), marker="^",
                   ls="None", lw=0, markersize=5.4,
                   label=rf"$\bar{{\Psi}}$ {sub_label}"),
        ]

    _add_cutoff_line(ax)
    _add_external_convergence_legend(
        fig,
        _beta_handles_for(plt.cm.plasma, "PBC", OMEGA_PBC),
        _beta_handles_for(plt.cm.viridis, "OBC", OMEGA_OBC),
        [
            _reference_handle(rf"theory $={BETA / NU:.4f}$"),
            _cutoff_handle(),
            *band_handles,
        ],
    )
    _finish_axis(fig, ax, r"$\beta/\nu$", "beta_over_nu_vs_Lmin.pdf",
                 build_legend=False)

    fig, ax = _new_axis()
    for primary, cmap, bc_label in specs:
        if not primary:
            continue
        _plot_block(ax, primary, cmap, bc_label, "z_dynamic_leading",
                    "_nolegend_", color_key="leading",
                    marker="s", alpha=0.78)
        if bc_label == "OBC":
            _plot_block(ax, primary, cmap, bc_label, "z_dynamic_mixed_subleading",
                        "_nolegend_",
                        color_key="effective", marker="D")
        else:
            _plot_block(ax, primary, cmap, bc_label, "z_dynamic_raw_subleading",
                        "_nolegend_", color_key="raw", marker="o")
    ax.axhline(Z_EXACT, color="gray", ls="--", lw=1.2, alpha=0.8,
               label="_nolegend_")
    pbc_val, pbc_drift = _final_estimate(primary_pbc, "z_dynamic_final")
    obc_val, obc_drift = _final_estimate(primary_obc, "z_dynamic_final")
    band_handles = [
        _numeric_band(ax, pbc_val, pbc_drift, color=plt.cm.plasma(0.83),
                      label="PBC Final"),
        _numeric_band(ax, obc_val, obc_drift, color=plt.cm.viridis(0.52),
                      label="OBC Final"),
    ]
    _add_cutoff_line(ax)
    _add_external_convergence_legend(
        fig,
        [
            _curve_handle(plt.cm.plasma, "leading", "s", "leading"),
            _curve_handle(plt.cm.plasma, "raw", "o",
                          rf"subl. $\omega={OMEGA_PBC:.0f}$"),
        ],
        [
            _curve_handle(plt.cm.viridis, "leading", "s", "leading"),
            _curve_handle(plt.cm.viridis, "effective", "D", "mixed subl."),
        ],
        [
            _reference_handle(rf"theory $={Z_EXACT:.4f}$"),
            _cutoff_handle(),
            *band_handles,
        ],
    )
    _finish_axis(fig, ax, r"$z$", "z_vs_Lmin.pdf", build_legend=False)

    fig, ax = _new_axis()
    for primary, cmap, bc_label in specs:
        if not primary:
            continue
        final_lmin = primary.get("final_lmin", FINAL_LMIN_DEFAULT)
        combos = [
            ("z_dynamic_leading", "beta_over_nu_leading",
             "_nolegend_", "gamma_leading", "s"),
        ]
        if bc_label == "OBC":
            combos.append(
                ("z_dynamic_mixed_subleading", "beta_over_nu_mixed_subleading",
                 "_nolegend_",
                 "gamma_effective", "D")
            )
        else:
            combos.append(
                ("z_dynamic_raw_subleading", "beta_over_nu_raw_subleading",
                 "_nolegend_", "gamma_raw", "o")
            )
        for z_key, beta_key, curve_label, color_key, marker in combos:
            if z_key not in primary or beta_key not in primary:
                continue
            sweep = gamma_over_nu_sweep_from_hyperscaling(primary, z_key, beta_key)
            _plot_sweep_axis(
                ax,
                sweep,
                color=cmap(color_pos[color_key]),
                label=curve_label,
                final_lmin=final_lmin,
                marker=marker,
                alpha=0.9,
            )
    ax.axhline(GAMMA / NU, color="gray", ls="--", lw=1.2, alpha=0.8,
               label="_nolegend_")
    pbc_val, pbc_drift = _gamma_final_estimate(primary_pbc)
    obc_val, obc_drift = _gamma_final_estimate(primary_obc)
    band_handles = [
        _numeric_band(ax, pbc_val, pbc_drift, color=plt.cm.plasma(0.83),
                      label="PBC Final"),
        _numeric_band(ax, obc_val, obc_drift, color=plt.cm.viridis(0.52),
                      label="OBC Final"),
    ]
    _add_cutoff_line(ax)
    _add_external_convergence_legend(
        fig,
        [
            _curve_handle(plt.cm.plasma, "gamma_leading", "s", "leading"),
            _curve_handle(plt.cm.plasma, "gamma_raw", "o",
                          rf"subl. $\omega={OMEGA_PBC:.0f}$"),
        ],
        [
            _curve_handle(plt.cm.viridis, "gamma_leading", "s", "leading"),
            _curve_handle(plt.cm.viridis, "gamma_effective", "D", "mixed subl."),
        ],
        [
            _reference_handle(rf"theory $={GAMMA / NU:.4f}$"),
            _cutoff_handle(),
            *band_handles,
        ],
    )
    _finish_axis(fig, ax, r"$\gamma/\nu$", "gamma_over_nu_vs_Lmin.pdf",
                 build_legend=False)


def plot_gap_derivative_nu(primary_pbc: dict, primary_obc: dict) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    def _plain_tick(value, _pos):
        if not np.isfinite(value) or value <= 0.0:
            return ""
        return np.format_float_positional(value, precision=4, trim="-")

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    cutoff_color = "#9a9a9a"
    legend_handles = []
    for primary, color, label in [
        (primary_pbc, plt.cm.plasma(0.62), "PBC"),
        (primary_obc, plt.cm.viridis(0.62), "OBC"),
    ]:
        if not primary:
            continue
        table = primary.get("gap_derivative", {})
        L = table.get("L", np.asarray([]))
        y = table.get("dlog", np.asarray([]))
        m = np.isfinite(L) & np.isfinite(y) & (L > 0.0) & (y > 0.0)
        if np.count_nonzero(m) == 0:
            continue
        ax.plot(L[m], y[m], ls="None", marker="o", color=color,
                label="_nolegend_")
        block = primary.get("nu_inv_final", primary.get("nu_inv", {}))
        final = _row_for_lmin(block, primary.get("final_lmin", FINAL_LMIN_DEFAULT))
        if final is not None and np.isfinite(final.get("A", np.nan)) and np.isfinite(final.get("value", np.nan)):
            mask = L >= float(final["Lmin"])
            if np.count_nonzero(mask & m) >= 2:
                L_fit = np.linspace(float(np.min(L[mask & m])), float(np.max(L[mask & m])), 200)
                y_fit = _eval_powerlaw_fit_row(L_fit, final, label, inverse=False)
                ax.plot(L_fit, y_fit, color=color, lw=1.4, ls="--",
                        label="_nolegend_")
        legend_handles.append(
            Line2D([], [], color=color, marker="o", ls="--", lw=1.4,
                   markersize=5.5, label=label)
        )

    ax.axvline(FINAL_LMIN_DEFAULT, color=cutoff_color, ls="--", lw=1.35,
               alpha=0.68)
    legend_handles.append(
        Line2D([], [], color=cutoff_color, ls="--", lw=1.35,
               label=r"$L^{\star}$")
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(_plain_tick))
    ax.yaxis.set_major_formatter(FuncFormatter(_plain_tick))
    ax.set_xlabel(r"$L$")
    ax.set_ylabel(r"$|\partial_g \ln\Delta_0|_{g_c}$")
    ax.tick_params(direction="in", which="both", top=True, right=True,
                   labelsize=TICK_FONTSIZE)
    apply_grid(ax)
    ax.legend(handles=legend_handles, frameon=False, fontsize=LEGEND_FONTSIZE)
    save_fig(fig, "gap_derivative_nu.pdf")


def _extract_gapL_at_gpc(
    gap_data: dict,
    g_pc: float,
    *,
    bc: str = "pbc",
    use_effective_length: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate Delta0(g,L)*scale at the approach pseudo-critical point."""
    L_vals, scale_vals, gapL_vals = [], [], []
    bc_key = str(bc).lower()
    for L in sorted(gap_data):
        gd = _gap_block(gap_data[L])
        cols = gap_columns(gd)
        g = np.asarray(cols["g_arr"], dtype=float)
        delta0 = np.asarray(cols["Delta0"], dtype=float)
        m = np.isfinite(g) & np.isfinite(delta0) & (delta0 > 0.0)
        if np.count_nonzero(m) < 4:
            continue
        g_use = g[m]
        if g_pc < np.min(g_use) or g_pc > np.max(g_use):
            continue
        scale = (
            float(effective_length(np.asarray([float(L)]), bc_key)[0])
            if use_effective_length else float(L)
        )
        spl = CubicSpline(g_use, delta0[m])
        val = float(spl(g_pc)) * scale
        if np.isfinite(val) and val > 0.0:
            L_vals.append(float(L))
            scale_vals.append(scale)
            gapL_vals.append(val)
    return (
        np.asarray(L_vals, dtype=float),
        np.asarray(scale_vals, dtype=float),
        np.asarray(gapL_vals, dtype=float),
    )


def extract_scaled_gap_at_gpc(
    gap_data: dict,
    bc: str,
    g_pc: float,
    use_effective_length: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return raw L, ell(L), and Delta0(g_pc,L)*ell(L).

    This deliberately interpolates the Delta0 column and then multiplies by
    ell. For OBC this avoids reusing the stored Delta0*L column when the
    spectral control needs Delta0*(L+1/2).
    """
    L_arr, ell_arr, scaled_gap = _extract_gapL_at_gpc(
        gap_data,
        g_pc,
        bc=bc,
        use_effective_length=use_effective_length,
    )
    return L_arr, ell_arr, scaled_gap


def _spectral_gap_amplitude_row(
    gap_data: dict,
    *,
    bc_label: str,
    g_pc: float,
    Lmin: int = FINAL_LMIN_DEFAULT,
    omega: float = GAP_RESIDUAL_OMEGA,
) -> dict:
    """Fit Delta0(g_pc,L)*ell = C*(1+B*ell^-omega) for one BC."""
    bc_key = bc_label.lower()
    use_effective_length = bc_key == "obc"
    L_arr, ell_arr, y_arr = extract_scaled_gap_at_gpc(
        gap_data,
        bc=bc_key,
        g_pc=g_pc,
        use_effective_length=use_effective_length,
    )
    m = (
        np.isfinite(L_arr)
        & np.isfinite(ell_arr)
        & np.isfinite(y_arr)
        & (ell_arr > 0.0)
        & (y_arr > 0.0)
        & (L_arr >= float(Lmin))
    )
    c_factor = 4.0 / np.pi if bc_key == "pbc" else 2.0 / np.pi
    C_theory = GAP_L_CFT_PBC if bc_key == "pbc" else GAP_L_CFT_OBC
    if np.count_nonzero(m) >= 3:
        fit = _fit_gap_L_fixed_omega(ell_arr[m], y_arr[m], omega)
        C_fit = float(fit["C"])
        B_fit = float(fit["B"])
        c_est = c_factor * C_fit if np.isfinite(C_fit) else np.nan
    else:
        C_fit = np.nan
        B_fit = np.nan
        c_est = np.nan
    return dict(
        BC=bc_label.upper(),
        Lmin=int(Lmin),
        npts=int(np.count_nonzero(m)),
        g_pc=float(g_pc),
        omega=float(omega),
        C=C_fit,
        B=B_fit,
        c_est=float(c_est) if np.isfinite(c_est) else np.nan,
        C_theory=float(C_theory),
        relerr_c_percent=100.0 * abs(c_est - C_EXACT) / C_EXACT if np.isfinite(c_est) else np.nan,
        L=L_arr,
        ell=ell_arr,
        scaled_gap=y_arr,
        fit_mask=m,
        cutoff_ell=float(effective_length(np.asarray([float(Lmin)]), bc_key)[0])
        if use_effective_length else float(Lmin),
    )


def _spectral_gap_amplitude_results(
    data_pbc: dict,
    data_obc: dict,
    *,
    g_pc_pbc: float,
    g_pc_obc: float,
) -> dict[str, dict]:
    rows = {
        "PBC": _spectral_gap_amplitude_row(
            data_pbc,
            bc_label="PBC",
            g_pc=g_pc_pbc,
            Lmin=FINAL_LMIN_DEFAULT,
            omega=GAP_RESIDUAL_OMEGA,
        ),
        "OBC": _spectral_gap_amplitude_row(
            data_obc,
            bc_label="OBC",
            g_pc=g_pc_obc,
            Lmin=FINAL_LMIN_DEFAULT,
            omega=GAP_RESIDUAL_OMEGA,
        ),
    }
    return rows


def _write_spectral_gap_amplitude_table(results: dict[str, dict]) -> None:
    path = FSS_DIR / "spectral_gap_amplitude_fit.dat"
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Spectral gap amplitude from Delta0(g_pc,L)*ell = C*(1+B*ell^{-omega})\n")
        f.write("# PBC: ell=L, c_est=4C/pi, C_theory=pi/2\n")
        f.write("# OBC: ell=L+1/2, c_est=2C/pi, C_theory=pi\n")
        f.write("# BC  Lmin  npts  g_pc  omega  C  B  c_est  C_theory  relerr_c_percent\n")
        for bc_label in ("PBC", "OBC"):
            row = results.get(bc_label, {})
            f.write(
                f"{bc_label:<3} {int(row.get('Lmin', FINAL_LMIN_DEFAULT)):5d} "
                f"{int(row.get('npts', 0)):5d} "
                f"{row.get('g_pc', np.nan):.12f} "
                f"{row.get('omega', np.nan):.6f} "
                f"{row.get('C', np.nan):.12e} "
                f"{row.get('B', np.nan):.12e} "
                f"{row.get('c_est', np.nan):.12e} "
                f"{row.get('C_theory', np.nan):.12e} "
                f"{row.get('relerr_c_percent', np.nan):.12e}\n"
            )
    print(f"  [OK] {path.name}")


def _draw_spectral_gap_amplitude_panel(ax: plt.Axes, row: dict, color) -> None:
    """Draw one BC panel for Delta0(g_pc,L)*ell versus ell."""
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    axis_label_fontsize = plt.rcParams["axes.labelsize"]
    tick_fontsize = plt.rcParams["xtick.labelsize"]
    title_fontsize = plt.rcParams["axes.titlesize"]
    legend_fontsize = plt.rcParams["legend.fontsize"]
    bc_label = row.get("BC", "")
    ell = np.asarray(row.get("ell", []), dtype=float)
    y = np.asarray(row.get("scaled_gap", []), dtype=float)
    fit_mask = np.asarray(row.get("fit_mask", np.zeros_like(ell, dtype=bool)), dtype=bool)
    finite = np.isfinite(ell) & np.isfinite(y) & (ell > 0.0) & (y > 0.0)
    L = np.asarray(row.get("L", []), dtype=float)
    if L.size != ell.size:
        L = np.full_like(ell, np.nan, dtype=float)
    low = finite & (L < 10.0)
    high = finite & (L >= 10.0)
    if np.count_nonzero(low):
        ax.plot(ell[low], y[low], marker="o", ls="None",
                markerfacecolor="none", markeredgecolor=color,
                markeredgewidth=1.1, markersize=3.4, label="_nolegend_")
    if np.count_nonzero(high):
        ax.plot(ell[high], y[high], marker="o", ls="None", color=color,
                markersize=3.4, label="_nolegend_")

    C_fit = float(row.get("C", np.nan))
    B_fit = float(row.get("B", np.nan))
    omega = float(row.get("omega", GAP_RESIDUAL_OMEGA))
    fit_handle = None
    if np.count_nonzero(fit_mask) >= 2 and np.isfinite(C_fit) and np.isfinite(B_fit):
        ell_fit_data = ell[fit_mask]
        ell_fit = np.geomspace(float(np.min(ell_fit_data)), float(np.max(ell_fit_data)), 300)
        y_fit = C_fit * (1.0 + B_fit * ell_fit ** (-omega))
        ax.plot(ell_fit, y_fit, color=color, lw=1.35, ls="-", label="_nolegend_")
        fit_handle = Line2D([], [], color=color, marker="o", ls="-", lw=1.35,
                            markersize=3.4, label="fit")

    C_theory = float(row.get("C_theory", np.nan))
    if np.isfinite(C_theory):
        theory_label = r"$\pi/2$" if bc_label == "PBC" else r"$\pi$"
        ax.axhline(C_theory, color="gray", ls="--", lw=1.25, alpha=0.8)
        if bc_label == "OBC":
            label_x = 0.02
            label_offset = (4, 5)
            label_ha = "left"
        else:
            label_x = 0.98
            label_offset = (-4, 5)
            label_ha = "right"
        ax.annotate(
            theory_label,
            xy=(label_x, C_theory),
            xycoords=("axes fraction", "data"),
            xytext=label_offset,
            textcoords="offset points",
            ha=label_ha,
            va="bottom",
            fontsize=axis_label_fontsize,
            color="0.35",
        )

    ax.set_xscale("log")
    ax.set_xlabel(
        r"$L$" if bc_label == "PBC" else r"$L_e=L+1/2$",
        fontsize=axis_label_fontsize,
    )
    ax.set_ylabel(
        r"$\Delta_0(g_{pc},L)\,L$"
        if bc_label == "PBC"
        else r"$\Delta_0(g_{pc},L_e)\,L_e$",
        fontsize=axis_label_fontsize,
    )
    ax.set_title(bc_label, fontsize=title_fontsize, loc="right")
    ax.tick_params(direction="in", which="both", top=True, right=True,
                   labelsize=tick_fontsize)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    apply_grid(ax)
    if fit_handle is not None:
        legend_loc = "upper right" if bc_label == "PBC" else "lower right"
        ax.legend(handles=[fit_handle], frameon=False,
                  fontsize=legend_fontsize, loc=legend_loc)


def extract_c_sweep(
    gap_data: dict,
    bc: str = "pbc",
    g_pc: float = G_C,
    *,
    approach: str = "subleading",
    use_effective_length: bool = False,
) -> list[dict]:
    """
    Fit Delta0(g_pc,L)*scale = A*(1+B*scale^{-omega}) and convert A to c.

    PBC: A = pi*c/4, so c = 4*A/pi.
    OBC: A = pi*c/2, so c = 2*A/pi.
    """
    bc_key = bc.lower()
    omega = GAP_RESIDUAL_OMEGA if use_effective_length else (OMEGA_PBC if bc_key == "pbc" else OBC_RAW_OMEGA)
    factor = 4.0 / np.pi if bc_key == "pbc" else 2.0 / np.pi
    L_arr, scale_arr, gapL_arr = _extract_gapL_at_gpc(
        gap_data,
        g_pc,
        bc=bc_key,
        use_effective_length=use_effective_length,
    )
    scale_label = "Le=L+1/2" if bc_key == "obc" and use_effective_length else "raw_L"
    rows = []
    for Lmin in LMIN_SWEEP:
        mask = L_arr >= float(Lmin)
        if np.count_nonzero(mask) < 3:
            continue
        res = _fit_gap_L_fixed_omega(scale_arr[mask], gapL_arr[mask], omega)
        amplitude = float(res["C"])
        rows.append(dict(
            Lmin=int(Lmin),
            n=int(np.count_nonzero(L_arr >= float(Lmin))),
            C=amplitude,
            B=float(res["B"]),
            omega=float(omega),
            factor=float(factor),
            g_pc=float(g_pc),
            scale=scale_label,
            approach=str(approach),
            use_effective_length=bool(use_effective_length),
            c=factor * amplitude if np.isfinite(amplitude) else np.nan,
        ))
    return rows


def _select_lmin_row(rows: list[dict], Lmin: int = FINAL_LMIN_DEFAULT) -> dict:
    valid = [r for r in rows if np.isfinite(r.get("C", np.nan))]
    row = next((r for r in valid if int(r["Lmin"]) == int(Lmin)), None)
    if row is None:
        row = next((r for r in valid if int(r["Lmin"]) >= int(Lmin)), None)
    if row is None and valid:
        row = valid[-1]
    if row is None:
        row = dict(Lmin=int(Lmin), n=0, C=np.nan, B=np.nan,
                   omega=np.nan, factor=np.nan, g_pc=np.nan, scale="", c=np.nan)
    return row


def _write_kink_velocity_sweep(results: dict) -> None:
    path = FSS_DIR / "kink_velocity_sweep.dat"
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Kink velocity c from Delta0(g_pc,L)*scale = A*(1+B*scale^{-omega})\n")
        f.write("# PBC: A = pi*c/4, c = 4*A/pi, omega = 2\n")
        f.write("# OBC mixed_subleading: scale=Le=L+1/2, A = pi*c/2, omega = 2\n")
        f.write("# BC approach Lmin n g_pc scale A B omega c\n")
        for bc_label in ("PBC", "OBC"):
            for row in results.get(bc_label, {}).get("sweep", []):
                approach = public_approach_label(bc_label, row.get("approach", ""))
                f.write(
                    f"{bc_label:<4} {approach:<28} {row['Lmin']:4d} {row['n']:3d} "
                    f"{row.get('g_pc', np.nan):.12f} "
                    f"{row.get('scale', ''):<10} "
                    f"{row['C']:.12e} {row['B']:.12e} "
                    f"{row['omega']:.6f} {row['c']:.12e}\n"
                )


def _draw_c_velocity_panel(ax: plt.Axes, row: dict, color) -> None:
    axis_label_fontsize = plt.rcParams["axes.labelsize"]
    tick_fontsize = plt.rcParams["xtick.labelsize"]
    title_fontsize = plt.rcParams["axes.titlesize"]
    bc_label = row.get("BC", "")
    rows = row.get("sweep", [])
    if rows:
        Lmins = np.asarray([r["Lmin"] for r in rows], dtype=float)
        c_vals = np.asarray([r["c"] for r in rows], dtype=float)
        m = np.isfinite(Lmins) & np.isfinite(c_vals)
        low = m & (Lmins < 10.0)
        high = m & (Lmins >= 10.0)
        if np.count_nonzero(low):
            ax.plot(Lmins[low], c_vals[low], marker="o", ls="None",
                    markerfacecolor="none", markeredgecolor=color,
                    markeredgewidth=1.1, markersize=3.4)
        if np.count_nonzero(high):
            ax.plot(Lmins[high], c_vals[high], marker="o", ls="None", color=color,
                    markersize=3.4)

    ax.axhline(C_EXACT, color="gray", ls=":", lw=1.2, alpha=0.8)
    ax.set_xlabel(r"$L_{\min}$", fontsize=axis_label_fontsize)
    ax.set_ylabel(r"$c$", fontsize=axis_label_fontsize)
    ax.set_xticks(LMIN_SWEEP)
    ax.set_title(bc_label, fontsize=title_fontsize, loc="right")
    ax.tick_params(direction="in", which="both", top=True, right=True,
                   labelsize=tick_fontsize)
    apply_grid(ax)


def plot_kink_velocity(
    data_pbc: dict,
    data_obc: dict,
    *,
    g_pc_pbc: float,
    g_pc_obc: float,
) -> dict:
    """Create kink_velocity_gap.pdf with c plateau and spectral gap-amplitude control."""
    sweep_pbc = extract_c_sweep(
        data_pbc,
        bc="pbc",
        g_pc=g_pc_pbc,
        approach="raw_subleading",
        use_effective_length=False,
    )
    sweep_obc = extract_c_sweep(
        data_obc,
        bc="obc",
        g_pc=g_pc_obc,
        approach="mixed_subleading",
        use_effective_length=True,
    )
    final_pbc = _select_lmin_row(sweep_pbc, FINAL_LMIN_DEFAULT)
    final_obc = _select_lmin_row(sweep_obc, FINAL_LMIN_DEFAULT)
    results = {
        "PBC": {**final_pbc, "sweep": sweep_pbc},
        "OBC": {**final_obc, "sweep": sweep_obc},
    }
    _write_kink_velocity_sweep(results)
    spectral_results = _spectral_gap_amplitude_results(
        data_pbc,
        data_obc,
        g_pc_pbc=g_pc_pbc,
        g_pc_obc=g_pc_obc,
    )
    _write_spectral_gap_amplitude_table(spectral_results)

    rc_params = {
        "font.size": 8.5,
        "axes.labelsize": 9.0,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 7.2,
    }
    with plt.rc_context(rc_params):
        fig, axes = plt.subplots(
            1, 4, figsize=(8.6, 2.25), constrained_layout=True
        )
        _draw_c_velocity_panel(axes[0], {**results["PBC"], "BC": "PBC"}, plt.cm.plasma(0.62))
        axes[0].set_ylim(1.996, 2.001)
        _draw_c_velocity_panel(axes[1], {**results["OBC"], "BC": "OBC"}, plt.cm.viridis(0.62))
        axes[1].set_ylabel("")
        axes[1].set_ylim(1.9995, 2.0020)
        _draw_spectral_gap_amplitude_panel(axes[2], spectral_results["PBC"], plt.cm.plasma(0.62))
        # Hide only the first numeric x-tick label on the third panel
        from matplotlib.ticker import FuncFormatter
        _xlim = axes[2].get_xlim()
        _xticks = axes[2].get_xticks()
        if _xticks.size:
            _first_tick = float(_xticks[0])
            _orig_formatter = axes[2].xaxis.get_major_formatter()

            def _third_panel_formatter(v, pos):
                if abs(float(v) - _first_tick) < 1e-12:
                    return ""
                try:
                    return _orig_formatter(v, pos)
                except Exception:
                    return f"{v:g}"

            axes[2].xaxis.set_major_formatter(FuncFormatter(_third_panel_formatter))
        axes[2].set_xlim(_xlim)
        axes[2].set_ylim(1.565, 1.595)
        _draw_spectral_gap_amplitude_panel(axes[3], spectral_results["OBC"], plt.cm.viridis(0.62))
        axes[3].set_ylim(3.120, 3.150)

    save_fig(fig, "kink_velocity_gap.pdf")
    return results


# ---------------------------------------------------------------------------
# Gap*L crossing plot + g_c estimate
# ---------------------------------------------------------------------------
def plot_gap_scaled_crossing(
    gap_data: dict[int, np.ndarray],
    gap_data_obc: dict[int, np.ndarray] | None = None,
    title_suffix: str = "",
    g_pc: float | None = None,
    g_pc_obc: float | None = None,
) -> None:
    """Plot scaled-gap crossings for PBC and OBC."""
    from matplotlib.lines import Line2D
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    _ = title_suffix
    datasets = [("PBC", gap_data, g_pc, GAP_L_CFT_PBC, r"$\pi/2$", plt.cm.plasma)]
    if gap_data_obc:
        datasets.append(("OBC", gap_data_obc, g_pc_obc, GAP_L_CFT_OBC, r"$\pi$", plt.cm.viridis))

    fig, axes = plt.subplots(
        1, len(datasets),
        figsize=(7 * len(datasets), 5),
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes.ravel()
    critical_line_style = dict(
        color="0.25",
        ls=(0, (1.2, 2.4)),
        lw=1.2,
        alpha=0.75,
        zorder=1,
    )
    pseudocritical_line_style = {
        **critical_line_style,
        "color": "tab:blue",
        "alpha": 0.85,
    }

    def _annotate_gc(ax_use):
        ax_use.axvline(G_C, **critical_line_style)

    plotted_L_methods: dict[int, set[int]] = {}
    plotted_methods: set[int] = set()

    def _draw_panel(ax_gapL, bc_label, data, g_pc_row, asymptote, asymptote_label, cmap):
        L_list = [L for L in sorted(data) if int(L) >= FINAL_LMIN_DEFAULT]
        colors = colors_for_bc(bc_label)
        crossing_inset_data = []

        ax_gapL.set_xlabel(r"$g$", fontsize=AXIS_LABEL_FONTSIZE)
        ax_gapL.tick_params(labelsize=TICK_FONTSIZE, direction="in", which="both", top=True, right=True)
        apply_grid(ax_gapL)
        ax_gapL.set_xlim(0.5, 1.5)
        ax_gapL.set_title(bc_label, fontsize=TITLE_FONTSIZE, loc="right")

        scaled_ylabel = r"$\Delta_0\,L_e$" if bc_label == "OBC" else r"$\Delta_0\,L$"
        ax_gapL.set_ylabel(scaled_ylabel, fontsize=AXIS_LABEL_FONTSIZE)
        _annotate_gc(ax_gapL)
        ax_gapL.axhline(asymptote, color="gray", ls=":", lw=1.4, alpha=0.7, zorder=1)
        ax_gapL.annotate(
            asymptote_label,
            xy=(1.5, asymptote),
            xycoords="data",
            xytext=(-6, 5),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=TEXT_FONTSIZE,
            color="gray",
        )

        for L in L_list:
            method = 0 if int(L) in ED_SIZES else 1
            ls = "-" if method == 0 else "--"
            c = colors.get(int(L), cmap(0.65))
            gd = data[L]
            gcols = gap_columns(gd)
            g = gcols["g_arr"]
            if bc_label == "OBC":
                scale = float(effective_length(np.asarray([float(L)]), "obc")[0])
                gap_scaled = gcols["Delta0"] * scale
            else:
                gap_scaled = gd[:, gcols["gapL_col"]]
            m = np.isfinite(g) & np.isfinite(gap_scaled)
            if np.count_nonzero(m) < 2:
                continue
            ax_gapL.plot(g[m], gap_scaled[m], ls=ls, color=c, lw=1.6,
                         label=rf"$L={L}$", zorder=2)
            _register_external_legend_item(plotted_L_methods, plotted_methods, int(L), method)
            crossing_inset_data.append((g[m], gap_scaled[m], c, ls))

        if crossing_inset_data:
            zoom_xlo, zoom_xhi = 0.9989, 1.001
            zoom_vals = []
            for g_use, gapL_use, _c, _ls in crossing_inset_data:
                mz = (g_use >= zoom_xlo) & (g_use <= zoom_xhi)
                if np.count_nonzero(mz) >= 2:
                    zoom_vals.extend(gapL_use[mz].tolist())

            if zoom_vals:
                axins = inset_axes(
                    ax_gapL,
                    width="45%",
                    height="45%",
                    loc="center left",
                    bbox_to_anchor=(0.12, 0.22, 1.0, 1.0),
                    bbox_transform=ax_gapL.transAxes,
                    borderpad=0.0,
                )
                for g_use, gapL_use, c, ls in crossing_inset_data:
                    axins.plot(g_use, gapL_use, ls=ls, color=c, lw=1.1)
                axins.axvline(G_C, **critical_line_style)
                if g_pc_row is not None and np.isfinite(g_pc_row):
                    axins.axvline(g_pc_row, **pseudocritical_line_style)
                axins.axhline(asymptote, color="gray", ls=":", lw=1.0, alpha=0.7)
                axins.set_xlim(zoom_xlo, zoom_xhi)
                zoom_arr = np.asarray(zoom_vals + [asymptote], dtype=float)
                zoom_arr = zoom_arr[np.isfinite(zoom_arr)]
                if bc_label == "OBC":
                    axins.set_ylim(3.07, 3.21)
                elif zoom_arr.size:
                    ylo, yhi = float(np.min(zoom_arr)), float(np.max(zoom_arr))
                    pad = 0.12 * max(yhi - ylo, 1e-6)
                    axins.set_ylim(ylo - pad, yhi + pad)
                if g_pc_row is not None and np.isfinite(g_pc_row):
                    axins.text(
                        g_pc_row,
                        0.92,
                        r"$g_{pc}$",
                        transform=axins.get_xaxis_transform(),
                        ha="right",
                        va="top",
                        fontsize=TEXT_FONTSIZE,
                        color="tab:blue",
                    )
                apply_two_decimal_y_ticks(axins)
                axins.tick_params(
                    labelsize=TICK_FONTSIZE,
                    direction="in",
                    which="both",
                    top=True,
                    right=True,
                )
                apply_grid(axins)

    for ax, args in zip(axes, datasets):
        _draw_panel(ax, *args)

    if axes.size >= 2:
        axes[1].set_ylim(axes[0].get_ylim())
        axes[1].tick_params(labelleft=False)

    size_handles = []
    size_labels = []
    for L in ALL_SIZES:
        if L not in plotted_L_methods:
            continue
        size_handles.append(_DoubleBCLine(L, _linestyle_for_method_set(plotted_L_methods[L])))
        size_labels.append(rf"$L={L}$")

    method_handles = []
    method_labels = []
    if 0 in plotted_methods:
        method_handles.append(Line2D([], [], color="gray", ls="-", lw=1.9))
        method_labels.append("ED")
    if 1 in plotted_methods:
        method_handles.append(Line2D([], [], color="gray", ls="--", lw=1.9))
        method_labels.append("LNCZ")

    legend_handles = size_handles + method_handles
    legend_labels = size_labels + method_labels
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.1 ),
            bbox_transform=fig.transFigure,
            frameon=False,
            fontsize=LEGEND_FONTSIZE,
            ncol=len(legend_handles),
            columnspacing=1.7,
            handlelength=2.2,
            handletextpad=0.8,
            handler_map={_DoubleBCLine: _DoubleBCLineHandler()},
        )
    save_fig(fig, "gap_scaled_crossing.pdf")


def estimate_gc_from_crossing(gap_data: dict[int, np.ndarray]) -> float:
    """Estimate g_c from intersections of gap*L curves for consecutive sizes."""
    L_list = sorted(gap_data)
    gc_vals = []
    gc_weights = []

    for k in range(len(L_list) - 1):
        L1, L2 = L_list[k], L_list[k + 1]
        d1, d2 = gap_data[L1], gap_data[L2]
        c1, c2 = gap_columns(d1), gap_columns(d2)

        g1, y1 = c1["g_arr"], d1[:, c1["gapL_col"]]
        g2, y2 = c2["g_arr"], d2[:, c2["gapL_col"]]

        m1 = np.isfinite(g1) & np.isfinite(y1)
        m2 = np.isfinite(g2) & np.isfinite(y2)
        g1, y1 = g1[m1], y1[m1]
        g2, y2 = g2[m2], y2[m2]
        if g1.size < 4 or g2.size < 4:
            continue

        g_min = max(float(np.min(g1)), float(np.min(g2)))
        g_max = min(float(np.max(g1)), float(np.max(g2)))
        if g_max <= g_min:
            continue

        g_grid = np.linspace(g_min, g_max, 2000)
        yy1 = np.interp(g_grid, g1, y1)
        yy2 = np.interp(g_grid, g2, y2)
        diff = yy1 - yy2

        sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
        for sc in sign_changes:
            g_lo, g_hi = g_grid[sc], g_grid[sc + 1]
            d_lo, d_hi = diff[sc], diff[sc + 1]
            if abs(d_hi - d_lo) < 1e-15:
                continue
            gc_est = g_lo - d_lo * (g_hi - g_lo) / (d_hi - d_lo)
            if np.isfinite(gc_est):
                gc_vals.append(float(gc_est))
                gc_weights.append(float(L1 * L2))

    if not gc_vals:
        return G_C

    gc_arr = np.asarray(gc_vals, dtype=float)
    wt_arr = np.asarray(gc_weights, dtype=float)
    return float(np.sum(wt_arr * gc_arr) / np.sum(wt_arr))


def _gap_block(entry):
    """Accept either raw gap array or (gap, obs, src) tuples."""
    if isinstance(entry, tuple):
        return entry[0]
    return entry


def plot_delta0_vs_L(data_pbc: dict, data_obc: dict) -> dict:
    """
    Diagnostic Delta_0(L) plots at fixed g for PBC/OBC.

    These plots do not define the final dynamic exponent used in the FSS
    summary. Each boundary-condition and scale view is saved separately.
    """
    from matplotlib.lines import Line2D

    G_TARGET = [0.75, 1.0, 1.2]
    L_FIT_MIN = 10.0
    cutoff_color = "#9a9a9a"
    markers = {0.75: "o", 1.0: "s", 1.2: "^"}
    _g_color_values = np.linspace(0.18, 0.86, len(G_TARGET))
    G_COLORS_BY_BC = {
        "PBC": {g: plt.cm.plasma(v) for g, v in zip(G_TARGET, _g_color_values)},
        "OBC": {g: plt.cm.viridis(v) for g, v in zip(G_TARGET, _g_color_values)},
    }

    def extract_delta_map(data: dict) -> dict:
        out = {}
        for g_target in G_TARGET:
            L_vals = []
            d_vals = []
            for L in sorted(data.keys()):
                gd = _gap_block(data[L])
                if gd.shape[1] < 4:
                    continue
                gcols = gap_columns(gd)
                x = gcols["g_arr"]
                y = gcols["Delta0"]
                if x.size < 4:
                    continue
                f = interp1d(x, y, kind="cubic", bounds_error=False,
                             fill_value=np.nan)
                d0g = float(f(g_target))
                if not np.isfinite(d0g) or d0g <= 0.0:
                    continue
                L_vals.append(float(L))
                d_vals.append(d0g)
            out[g_target] = {
                "L": np.asarray(L_vals, dtype=float),
                "D0": np.asarray(d_vals, dtype=float),
            }
        return out

    dmap_pbc = extract_delta_map(data_pbc)
    dmap_obc = extract_delta_map(data_obc)

    panel_spec = [
        (dmap_pbc, "PBC", "semilog", "pbc_semilog.pdf"),
        (dmap_pbc, "PBC", "loglog", "pbc_loglog.pdf"),
        (dmap_obc, "OBC", "semilog", "obc_semilog.pdf"),
        (dmap_obc, "OBC", "loglog", "obc_loglog.pdf"),
    ]

    gap_source = {"PBC": data_pbc, "OBC": data_obc}
    fit_summary = {
        "PBC": {"g": np.nan, "z": np.nan, "omega": np.nan, "C": np.nan, "B": np.nan},
        "OBC": {"g": np.nan, "z": np.nan, "omega": np.nan, "C": np.nan, "B": np.nan},
    }

    for dmap, bc_label, scale_mode, out_name in panel_spec:
        fig, ax = plt.subplots(figsize=(6.2, 5.0), constrained_layout=True)
        g_colors = G_COLORS_BY_BC[bc_label]
        for g in G_TARGET:
            L_arr = dmap[g]["L"]
            D0_arr = dmap[g]["D0"]
            if L_arr.size == 0:
                continue

            ax.plot(L_arr, D0_arr, ls="None", marker=markers[g], markersize=6,
                    color=g_colors[g], label=rf"$g={g}$", zorder=3)

            mask = L_arr >= L_FIT_MIN
            if np.count_nonzero(mask) < 3:
                mask = np.ones_like(L_arr, dtype=bool)
            if np.count_nonzero(mask) < 3:
                continue

            if g != 1.0:
                coeffs = np.polyfit(L_arr[mask], np.log(D0_arr[mask]), 1)
                L_fit = np.linspace(L_arr[mask].min(), L_arr[mask].max(), 200)
                D0_fit = np.exp(np.polyval(coeffs, L_fit))
            else:
                coeffs = np.polyfit(np.log(L_arr[mask]), np.log(D0_arr[mask]), 1)
                L_fit = np.linspace(L_arr[mask].min(), L_arr[mask].max(), 200)
                D0_fit = np.exp(np.polyval(coeffs, np.log(L_fit)))

            ax.plot(L_fit, D0_fit, ls="--", lw=1.5, color=g_colors[g], zorder=2)

            if g == 1.0 and scale_mode == "loglog":
                fit_summary[bc_label]["g"] = float(g)
                fit_summary[bc_label]["z"] = float(-coeffs[0])

                L_gc = []
                gapL_gc = []
                for L in sorted(gap_source[bc_label].keys()):
                    gd = _gap_block(gap_source[bc_label][L])
                    if gd.shape[1] < 6:
                        continue
                    gcols = gap_columns(gd)
                    f_gapL = interp1d(gcols["g_arr"], gd[:, gcols["gapL_col"]], kind="cubic",
                                      bounds_error=False, fill_value=np.nan)
                    val = float(f_gapL(1.0))
                    if np.isfinite(val) and val > 0.0:
                        L_gc.append(float(L))
                        gapL_gc.append(val)

                L_gc = np.asarray(L_gc, dtype=float)
                gapL_gc = np.asarray(gapL_gc, dtype=float)
                if L_gc.size >= 3:
                    omega_fixed = OMEGA_PBC if bc_label == "PBC" else OMEGA_OBC
                    res = _fit_gap_L_fixed_omega(
                        np.asarray(L_gc, dtype=float),
                        np.asarray(gapL_gc, dtype=float),
                        omega=omega_fixed,
                    )
                    omega_val = omega_fixed            # fixed from theory
                    fit_summary[bc_label]["omega"] = float(omega_val)
                    fit_summary[bc_label]["C"] = float(res["C"])
                    fit_summary[bc_label]["B"] = float(res["B"])

        if scale_mode == "semilog":
            ax.set_yscale("log")
        else:
            ax.set_xscale("log")
            ax.set_yscale("log")

        ax.set_xlabel(r"$L$", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(r"$\Delta_0$", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_title(bc_label, loc="right", fontsize=TITLE_FONTSIZE)
        ax.axvline(L_FIT_MIN, color=cutoff_color, ls="--", lw=1.35, alpha=0.68)
        ax.tick_params(direction="in", which="both", top=True, right=True,
                       labelsize=TICK_FONTSIZE)
        apply_grid(ax)

        leg_handles = [
            Line2D([], [], color=G_COLORS_BY_BC[bc_label][g],
                   marker=markers[g], lw=0, markersize=7,
                   label=rf"$g={g}$")
            for g in G_TARGET
        ]
        leg_handles.append(
            Line2D([], [], color=cutoff_color, ls="--", lw=1.35,
                   label=r"$L^{\star}$")
        )
        ax.legend(handles=leg_handles, frameon=False, fontsize=LEGEND_FONTSIZE, loc="best")

        save_fig(fig, f"delta0_vs_L/{out_name}")
    return fit_summary


def plot_gapL_subleading(
    data_pbc: dict,
    data_obc: dict,
    *,
    g_pc_pbc: float = GC_EXACT,
    g_pc_obc: float = GC_EXACT,
) -> dict:
    """Plot fixed-omega corrections of Delta_0 ell near the critical point."""
    FIT_LMIN = 10.0
    ASYMP = {"PBC": GAP_L_CFT_PBC, "OBC": GAP_L_CFT_OBC}
    OMEGA = {"PBC": OMEGA_PBC, "OBC": GAP_RESIDUAL_OMEGA}
    GPC = {"PBC": g_pc_pbc, "OBC": g_pc_obc}
    datasets = {"PBC": data_pbc, "OBC": data_obc}

    gapL_map: dict = {}
    for bc_label, data in datasets.items():
        L_vals, ell_vals, gapL_vals = [], [], []
        for L in sorted(data.keys()):
            gd = _gap_block(data[L])
            if gd.shape[1] < 5:
                continue
            gcols = gap_columns(gd)
            ell = (
                float(effective_length(np.asarray([float(L)]), "obc")[0])
                if bc_label == "OBC" else float(L)
            )
            f = interp1d(gcols["g_arr"], gcols["Delta0"],
                         kind="cubic", bounds_error=False, fill_value=np.nan)
            val = float(f(GPC[bc_label])) * ell
            if np.isfinite(val) and val > 0.0:
                L_vals.append(float(L))
                ell_vals.append(float(ell))
                gapL_vals.append(val)
        gapL_map[bc_label] = {
            "L": np.asarray(L_vals, dtype=float),
            "ell": np.asarray(ell_vals, dtype=float),
            "gapL": np.asarray(gapL_vals, dtype=float),
        }

    fit_results: dict = {}
    for bc_label in ("PBC", "OBC"):
        L_arr = gapL_map[bc_label]["L"]
        ell_arr = gapL_map[bc_label]["ell"]
        gapL_arr = gapL_map[bc_label]["gapL"]
        fit_mask = L_arr >= FIT_LMIN
        if np.count_nonzero(fit_mask) < 3:
            fit_results[bc_label] = {}
            continue
        omega = OMEGA[bc_label]
        fit_results[bc_label] = _fit_gap_L_fixed_omega(
            ell_arr[fit_mask], gapL_arr[fit_mask], omega
        )

    fig_a, axes_a = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    from matplotlib.lines import Line2D

    for ax, bc_label in zip(axes_a, ("PBC", "OBC")):
        L_arr = gapL_map[bc_label]["L"]
        ell_arr = gapL_map[bc_label]["ell"]
        gapL_arr = gapL_map[bc_label]["gapL"]
        asymp = ASYMP[bc_label]
        res = fit_results.get(bc_label, {})
        panel_color = plt.cm.plasma(0.62) if bc_label == "PBC" else plt.cm.viridis(0.62)
        asymp_label = r"$\pi/2$" if bc_label == "PBC" else r"$\pi$"
        legend_loc = "upper right"
        legend_handles = []
        scale_tex = "L" if bc_label == "PBC" else "L_E"

        if L_arr.size > 0:
            ax.scatter(ell_arr, gapL_arr, zorder=4, color=panel_color,
                       marker="o", s=35)
            legend_handles.append(
                Line2D([], [], color=panel_color, marker="o", lw=0,
                       markersize=6, label=rf"$\Delta_0 {scale_tex}$")
            )
        ax.axhline(asymp, color="gray", ls=":", lw=1.5, alpha=0.7)
        asymp_x = 0.02 if bc_label == "PBC" else 0.98
        asymp_ha = "left" if bc_label == "PBC" else "right"
        asymp_offset = (4, 4) if bc_label == "PBC" else (-4, 4)
        ax.annotate(
            asymp_label,
            xy=(asymp_x, asymp),
            xycoords=ax.get_yaxis_transform(),
            xytext=asymp_offset,
            textcoords="offset points",
            ha=asymp_ha,
            va="bottom",
            fontsize=TEXT_FONTSIZE,
            color="gray",
        )

        fit_mask = L_arr >= FIT_LMIN
        if res and np.count_nonzero(fit_mask) >= 3:
            C_hat = res["C"]
            B_hat = res["B"]
            omega = res["omega"]
            ell_fit = ell_arr[fit_mask]
            ell_fine = np.linspace(ell_fit.min(), ell_fit.max() * 1.1, 300)
            fit_curve = C_hat * (1.0 + B_hat * ell_fine ** (-omega))
            ax.plot(ell_fine, fit_curve, color="black", ls="--", lw=1.5)
            legend_handles.extend([
                Line2D([], [], color="black", lw=1.5, ls="--",
                       label=rf"fit: $C(1+B{scale_tex}^{{-{omega:.0f}}})$"),
                Line2D([], [], color="none", label=rf"$C={C_hat:.6f}$"),
                Line2D([], [], color="none", label=rf"$B={B_hat:.4f}$"),
            ])

        ax.set_xscale("log")
        if bc_label == "PBC":
            ax.set_ylim(bottom=1.568)
        else:
            ax.set_ylim(top=3.18)
        ax.set_xlabel(rf"${scale_tex}$", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(rf"$\Delta_0 {scale_tex}$", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_title(bc_label, fontsize=TITLE_FONTSIZE, loc="right")
        ax.tick_params(direction="in", which="both", top=True, right=True)
        apply_grid(ax)
        leg = ax.legend(handles=legend_handles, frameon=True, facecolor="white",
                        edgecolor="none", framealpha=1.0, fontsize=LEGEND_FONTSIZE,
                        loc=legend_loc, ncol=1)
        leg.get_frame().set_linewidth(0.0)

    save_fig(fig_a, "gapL_vs_L.pdf")

    return fit_results


def chiperp_critical_table(obs_data, g_pc, use_g_chi: bool = True) -> dict[str, np.ndarray]:
    """Interpolate chi_x(L,g) at g_pc and return C_x=g_pc*chi_x."""
    _ = use_g_chi
    L_vals, chi_vals, cperp_vals = [], [], []
    if not np.isfinite(g_pc):
        return dict(
            L=np.asarray([], dtype=float),
            chi_perp=np.asarray([], dtype=float),
            C_perp=np.asarray([], dtype=float),
        )

    for L in sorted(obs_data.keys()):
        g_arr, od = obs_data[L]
        if od.shape[1] <= COL["chi_perp"]:
            continue
        chi_col = np.asarray(od[:, COL["chi_perp"]], dtype=float)
        g_col = np.asarray(g_arr, dtype=float)
        m = np.isfinite(g_col) & np.isfinite(chi_col)
        if np.count_nonzero(m) < 4:
            continue

        g_use = g_col[m]
        chi_use = chi_col[m]
        order = np.argsort(g_use)
        g_use = g_use[order]
        chi_use = chi_use[order]
        g_use, unique_idx = np.unique(g_use, return_index=True)
        chi_use = chi_use[unique_idx]
        if g_use.size < 4 or g_pc < float(np.min(g_use)) or g_pc > float(np.max(g_use)):
            continue

        try:
            spl = CubicSpline(g_use, chi_use)
            chi_pc = float(spl(g_pc))
        except (ValueError, FloatingPointError):
            continue
        c_perp = float(g_pc) * chi_pc
        if np.isfinite(chi_pc) and chi_pc > 0.0 and np.isfinite(c_perp) and c_perp > 0.0:
            L_vals.append(float(L))
            chi_vals.append(chi_pc)
            cperp_vals.append(c_perp)

    return dict(
        L=np.asarray(L_vals, dtype=float),
        chi_perp=np.asarray(chi_vals, dtype=float),
        C_perp=np.asarray(cperp_vals, dtype=float),
    )


def _nan_chiperp_log_row(
    Lmin: int | float,
    *,
    n: int = 0,
    fit_type: str = "log_leading",
    omega: float = np.nan,
) -> dict:
    return dict(
        Lmin=int(Lmin),
        n=int(n),
        a0=np.nan,
        a1=np.nan,
        b=np.nan,
        omega=float(omega) if np.isfinite(omega) else np.nan,
        resid_rms=np.nan,
        fit_type=fit_type,
        alpha_log=0.0,
        role="logarithmic_alpha_zero",
        value=np.nan,
    )


def fit_chiperp_log_law(L, y, Lmin) -> dict:
    """Fit C_x(L)=a0+a1 log L on raw L."""
    L_arr = np.asarray(L, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    m = np.isfinite(L_arr) & np.isfinite(y_arr) & (L_arr > 0.0) & (L_arr >= float(Lmin))
    L_use = L_arr[m]
    y_use = y_arr[m]
    if L_use.size:
        idx = np.argsort(L_use)
        L_use = L_use[idx]
        y_use = y_use[idx]
    if L_use.size < 2:
        return _nan_chiperp_log_row(Lmin, n=L_use.size, fit_type="log_leading")

    try:
        xmat = np.column_stack([np.ones_like(L_use), np.log(L_use)])
        coeff = np.linalg.lstsq(xmat, y_use, rcond=None)[0]
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return _nan_chiperp_log_row(Lmin, n=L_use.size, fit_type="log_leading")

    a0, a1 = (float(coeff[0]), float(coeff[1]))
    y_fit = a0 + a1 * np.log(L_use)
    resid = y_use - y_fit
    return dict(
        Lmin=int(Lmin),
        n=int(L_use.size),
        a0=a0,
        a1=a1,
        b=np.nan,
        omega=np.nan,
        resid_rms=float(np.sqrt(np.mean(resid ** 2))) if resid.size else np.nan,
        fit_type="log_leading",
        alpha_log=0.0,
        role="logarithmic_alpha_zero",
        value=a1,
    )


def fit_chiperp_log_subleading(L, y, Lmin, omega) -> dict:
    """Fit C_x(L)=a0+a1 log L+b L^-omega on raw L."""
    L_arr = np.asarray(L, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    omega = float(omega)
    m = (
        np.isfinite(L_arr)
        & np.isfinite(y_arr)
        & (L_arr > 0.0)
        & (L_arr >= float(Lmin))
    )
    L_use = L_arr[m]
    y_use = y_arr[m]
    if L_use.size:
        idx = np.argsort(L_use)
        L_use = L_use[idx]
        y_use = y_use[idx]
    if L_use.size < 3 or not np.isfinite(omega):
        return _nan_chiperp_log_row(
            Lmin, n=L_use.size, fit_type="log_subleading", omega=omega
        )

    try:
        xmat = np.column_stack([
            np.ones_like(L_use),
            np.log(L_use),
            L_use ** (-omega),
        ])
        coeff = np.linalg.lstsq(xmat, y_use, rcond=None)[0]
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return _nan_chiperp_log_row(
            Lmin, n=L_use.size, fit_type="log_subleading", omega=omega
        )

    a0, a1, b = (float(coeff[0]), float(coeff[1]), float(coeff[2]))
    y_fit = a0 + a1 * np.log(L_use) + b * L_use ** (-omega)
    resid = y_use - y_fit
    return dict(
        Lmin=int(Lmin),
        n=int(L_use.size),
        a0=a0,
        a1=a1,
        b=b,
        omega=omega,
        resid_rms=float(np.sqrt(np.mean(resid ** 2))) if resid.size else np.nan,
        fit_type="log_subleading",
        alpha_log=0.0,
        role="logarithmic_alpha_zero",
        value=a1,
    )


def chiperp_log_sweep(L, y, bc, lmins: list[int] = LMIN_SWEEP) -> dict:
    """Return leading and fixed-omega logarithmic sweeps for C_perp."""
    bc_key = str(bc).lower()
    omega = OMEGA_PBC if bc_key == "pbc" else OMEGA_OBC
    leading = []
    subleading = []
    for Lmin in lmins:
        leading.append(fit_chiperp_log_law(L, y, Lmin))
        subleading.append(fit_chiperp_log_subleading(L, y, Lmin, omega))
    return dict(
        leading=leading,
        subleading=subleading,
        omega_subleading=float(omega),
        scale="raw_L",
        observable="C_perp",
        role="logarithmic_alpha_zero",
        diagnostic_only=True,
    )


def plot_fss_collapse(obs_pbc, obs_obc, g_c_num,
                      nu_num, beta_over_nu, gamma_over_nu,
                      g_c_num_obc=None, nu_num_obc=None,
                      beta_over_nu_obc=None):
    _ = gamma_over_nu  # chi_z collapse is handled from FD files only.
    if g_c_num_obc is None:
        g_c_num_obc = g_c_num
    if nu_num_obc is None:
        nu_num_obc = nu_num
    if beta_over_nu_obc is None or not np.isfinite(beta_over_nu_obc):
        beta_over_nu_obc = beta_over_nu
    from matplotlib.lines import Line2D
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    plotted_L_methods: dict[int, set[int]] = {}
    plotted_methods: set[int] = set()
    axes_by_panel = {
        (0, 0): axes[0, 0],
        (0, 1): axes[1, 0],
        (1, 0): axes[0, 1],
        (1, 1): axes[1, 1],
    }
    panel_data = [("PBC", obs_pbc, 0), ("OBC", obs_obc, 1)]
    psi_cols = [
        ("psi_tilde", COL["psi_tilde"], r"$\tilde{\Psi}\,L^{\beta/\nu}$"),
        ("psi_bar", COL["psi_bar"], r"$\bar{\Psi}\,L^{\beta/\nu}$"),
    ]
    for bc_label, obs_data, row in panel_data:
        g_ref = g_c_num if bc_label == "PBC" else g_c_num_obc
        nu_ref = nu_num if bc_label == "PBC" else nu_num_obc
        beta_ref = beta_over_nu if bc_label == "PBC" else beta_over_nu_obc
        Ls = sorted(obs_data.keys())
        if not Ls:
            continue
        colors = colors_for_bc(bc_label)
        for col_idx_panel, (_name, col, ylabel) in enumerate(psi_cols):
            ax = axes_by_panel[(row, col_idx_panel)]
            swap_inset_legend = col_idx_panel == 1
            inset_loc = "upper right" if swap_inset_legend else "lower left"
            inset_anchor = (
                (0.0, 0.0, 1.0, 1.0)
                if swap_inset_legend
                else (0.07, 0.05, 1.0, 1.0)
            )
            axins = inset_axes(
                ax,
                width="34%",
                height="34%",
                loc=inset_loc,
                bbox_to_anchor=inset_anchor,
                bbox_transform=ax.transAxes,
                borderpad=1.0,
            )
            zoom_y_vals = []
            size_handles: dict[int, Line2D] = {}
            for L in Ls:
                if int(L) < 10:
                    continue
                c = colors.get(L, "gray")
                method = 0 if int(L) in ED_SIZES else 1
                ls = "-" if method == 0 else "--"
                g_arr, od = obs_data[L]
                if od.shape[1] <= col:
                    continue
                x_res = (g_arr - g_ref) * (L ** (1.0 / nu_ref))
                y_res = od[:, col] * (L ** beta_ref)
                m = np.isfinite(x_res) & np.isfinite(y_res) & (y_res > 0.0)
                if np.count_nonzero(m) < 2:
                    continue
                line, = ax.plot(x_res[m], y_res[m], color=c, ls=ls, lw=1.2,
                                label=rf"$L={L}$")
                size_handles[int(L)] = line
                _register_external_legend_item(plotted_L_methods, plotted_methods, int(L), method)
                mz = m & (x_res >= -0.2) & (x_res <= 0.2)
                if np.count_nonzero(mz) >= 2:
                    axins.plot(x_res[mz], y_res[mz], color=c, ls=ls, lw=1.0)
                    zoom_y_vals.extend(y_res[mz].tolist())

            ax.axvline(0.0, color="gray", ls=":", alpha=0.5)
            ax.set_xlim(-6, 6)
            ax.set_xlabel(r"$(g - g_{pc})\,L^{1/\nu}$")
            ax.set_ylabel(ylabel)
            ax.tick_params(direction="in", which="both", top=True, right=True,
                           labelsize=TICK_FONTSIZE)
            apply_grid(ax)
            ax.set_title(bc_label, loc="right", fontsize=TITLE_FONTSIZE)
            axins.axvline(0.0, color="gray", ls=":", lw=1.0, alpha=0.6)
            axins.set_xlim(-0.2, 0.2)
            zoom_y = np.asarray(zoom_y_vals, dtype=float)
            zoom_y = zoom_y[np.isfinite(zoom_y)]
            if zoom_y.size:
                ylo = float(np.min(zoom_y))
                yhi = float(np.max(zoom_y))
                if yhi > ylo:
                    pad = 0.08 * (yhi - ylo)
                    axins.set_ylim(ylo - pad, yhi + pad)
            apply_two_decimal_y_ticks(axins)
            axins.tick_params(direction="in", which="both", top=True, right=True,
                              labelsize=TICK_FONTSIZE)
            apply_grid(axins)

    _external_size_method_legend(fig, plotted_L_methods, plotted_methods)
    save_psi_fig(fig, "psi_fss.pdf")


def plot_binder_collapse(obs_pbc, obs_obc, g_c_num, nu_num,
                         g_c_num_obc=None, nu_num_obc=None) -> None:
    """FSS collapse of the Binder cumulant for PBC/OBC."""
    if g_c_num_obc is None:
        g_c_num_obc = g_c_num
    if nu_num_obc is None:
        nu_num_obc = nu_num
    from matplotlib.lines import Line2D
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.03, wspace=0.03, hspace=0.02)
    plotted_L_methods: dict[int, set[int]] = {}
    plotted_methods: set[int] = set()
    for ax, bc_label, obs_data in [
        (axes[0], "PBC", obs_pbc),
        (axes[1], "OBC", obs_obc),
    ]:
        g_ref = g_c_num if bc_label == "PBC" else g_c_num_obc
        nu_ref = nu_num if bc_label == "PBC" else nu_num_obc
        Ls = sorted(obs_data.keys())
        if not Ls:
            continue
        colors = colors_for_bc(bc_label)
        axins = inset_axes(
            ax,
            width="40%",
            height="40%",
            loc="upper left",
            bbox_to_anchor=(0.08, 0.0, 1.0, 1.0),
            bbox_transform=ax.transAxes,
            borderpad=0.8,
        )
        zoom_y_vals = []
        L_plot = [int(L) for L in Ls if int(L) >= 10]
        if L_plot and np.isfinite(nu_ref) and nu_ref > 0.0:
            x_half = 0.5 * (min(L_plot) ** (1.0 / nu_ref))
        else:
            x_half = 6.0
        for L in Ls:
            if int(L) < 10:
                continue
            c = colors.get(L, "gray")
            method = 0 if int(L) in ED_SIZES else 1
            ls = "-" if method == 0 else "--"
            g_arr, od = obs_data[L]
            if od.shape[1] <= COL["binder"]:
                continue
            u = od[:, COL["binder"]]
            x_res = (g_arr - g_ref) * (L ** (1.0 / nu_ref))
            y_res = u
            m = np.isfinite(x_res) & np.isfinite(y_res)
            if np.count_nonzero(m) < 2:
                continue
            ax.plot(x_res[m], y_res[m], color=c, ls=ls, lw=1.2,
                    label=rf"$L={L}$")
            _register_external_legend_item(plotted_L_methods, plotted_methods, int(L), method)
            mz = m & (x_res >= -0.2) & (x_res <= 0.2)
            if np.count_nonzero(mz) >= 2:
                axins.plot(x_res[mz], y_res[mz], color=c, ls=ls, lw=1.0)
                zoom_y_vals.extend(y_res[mz].tolist())

        ax.axvline(0.0, color="gray", ls=":", alpha=0.5)
        ax.set_xlim(-x_half, x_half)
        ax.set_xlabel(r"$(g - g_c)\,L^{1/\nu}$")
        ax.set_ylabel(r"$U_4$" if bc_label == "PBC" else "")
        ax.set_title(bc_label, loc="right", fontsize=TITLE_FONTSIZE)
        ax.tick_params(direction="in", which="both", top=True, right=True,
                       labelsize=TICK_FONTSIZE)
        apply_grid(ax)
        axins.axvline(0.0, color="gray", ls=":", lw=1.0, alpha=0.6)
        axins.set_xlim(-0.2, 0.2)
        zoom_y = np.asarray(zoom_y_vals, dtype=float)
        zoom_y = zoom_y[np.isfinite(zoom_y)]
        if zoom_y.size:
            ylo = float(np.min(zoom_y))
            yhi = float(np.max(zoom_y))
            if yhi > ylo:
                pad = 0.08 * (yhi - ylo)
                axins.set_ylim(ylo - pad, yhi + pad)
        apply_two_decimal_y_ticks(axins)
        axins.tick_params(direction="in", which="both", top=True, right=True,
                          labelsize=TICK_FONTSIZE)
        apply_grid(axins)

    axes[1].set_ylim(axes[0].get_ylim())
    axes[1].tick_params(labelleft=False)

    _external_size_method_legend(fig, plotted_L_methods, plotted_methods)
    save_fig(fig, "binder_collapse.pdf")




def plot_psi_fss(obs_pbc, obs_obc, g_c_num, nu_num,
                 beta_ov_nu, primary_pbc: dict | None = None,
                 primary_obc: dict | None = None) -> None:
    g_c_num_obc = (
        (primary_obc or {}).get("beta_over_nu_final", {}).get("g_pc", np.nan)
        if primary_obc else np.nan
    )
    if not np.isfinite(g_c_num_obc):
        g_c_num_obc = g_c_num
    beta_ov_nu_obc = (
        (primary_obc or {}).get("beta_over_nu_final", {}).get("value", np.nan)
        if primary_obc else np.nan
    )
    if not np.isfinite(beta_ov_nu_obc):
        beta_ov_nu_obc = beta_ov_nu
    nu_num_obc = np.nan
    if primary_obc:
        inv_obc = primary_obc.get("nu_inv_final", {}).get("value", np.nan)
        if np.isfinite(inv_obc) and inv_obc != 0.0:
            nu_num_obc = 1.0 / inv_obc
    if not np.isfinite(nu_num_obc):
        nu_num_obc = nu_num
    psi_tables_by_bc = {
        "PBC": psifss_tables(obs_pbc, g_c_num),
        "OBC": psifss_tables(obs_obc, g_c_num_obc),
    }
    final_lmins = {
        "PBC": (primary_pbc or {}).get("final_lmin", FINAL_LMIN_DEFAULT),
        "OBC": (primary_obc or {}).get("final_lmin", FINAL_LMIN_DEFAULT),
    }
    primary_by_bc = {
        "PBC": primary_pbc or {},
        "OBC": primary_obc or {},
    }

    def _plot_psi_gc_fit(field: str, ylabel: str, filename: str) -> None:
        from matplotlib.lines import Line2D

        def _fit_context(bc_label: str):
            tab = psi_tables_by_bc[bc_label]
            L_arr = np.asarray(tab["L"], dtype=float)
            Y_arr = np.asarray(tab[field], dtype=float)
            primary = primary_by_bc[bc_label]
            block_key = "beta_over_nu_bar_final" if field == "psiB" else "beta_over_nu_final"
            block = primary.get(block_key, {})
            sweep = block.get("sweep", _sweep_powerlaw(L_arr, Y_arr, inverse=True))
            final_lmin = int(final_lmins[bc_label])
            final = next((r for r in sweep if int(r["Lmin"]) == final_lmin), None)
            if final is None:
                final = next((r for r in sweep if int(r["Lmin"]) >= final_lmin), None)
            if final is None and sweep:
                final = sweep[-1]
            beta_fit = final["value"] if final else np.nan
            return L_arr, Y_arr, sweep, final, beta_fit

        if field == "psiB":
            fig2, (ax_beta, ax_psi) = plt.subplots(
                1, 2, figsize=(13, 5), constrained_layout=True
            )
            cutoff_color = "#9a9a9a"
            cutoff_handle = Line2D([], [], color=cutoff_color, ls="--", lw=1.35,
                                   label=r"$L^{\star}$")
            psi_handles = []
            specs = [
                ("PBC", plt.cm.plasma(0.62), "o"),
                ("OBC", plt.cm.viridis(0.62), "s"),
            ]

            for bc_label, color, marker in specs:
                L_arr, Y_arr, sweep, final, beta_fit = _fit_context(bc_label)
                plotted_psi = False
                if sweep:
                    Lmins = np.asarray([r["Lmin"] for r in sweep], dtype=float)
                    beta_vals = np.asarray([r["value"] for r in sweep], dtype=float)
                    m = np.isfinite(Lmins) & np.isfinite(beta_vals)
                    if np.count_nonzero(m):
                        ax_beta.plot(
                            Lmins[m],
                            beta_vals[m],
                            marker=marker,
                            ls="-",
                            lw=1.5,
                            color=color,
                            label=bc_label,
                        )
                if final is not None and np.isfinite(beta_fit):
                    ax_beta.annotate(
                        rf"{bc_label}: {beta_fit:.4f}",
                        xy=(float(final["Lmin"]), beta_fit),
                        xytext=(5, 5 if bc_label == "PBC" else -14),
                        textcoords="offset points",
                        color=color,
                        fontsize=TEXT_FONTSIZE,
                    )

                m_data = np.isfinite(L_arr) & np.isfinite(Y_arr) & (L_arr > 0.0) & (Y_arr > 0.0)
                if np.count_nonzero(m_data):
                    ax_psi.plot(
                        L_arr[m_data],
                        Y_arr[m_data],
                        ls="None",
                        marker=marker,
                        color=color,
                        label="_nolegend_",
                    )
                    plotted_psi = True

                if (
                    final is not None
                    and np.isfinite(final.get("A", np.nan))
                    and np.isfinite(beta_fit)
                    and np.count_nonzero(m_data)
                ):
                    mask = m_data & (L_arr >= float(final["Lmin"]))
                    if np.count_nonzero(mask) >= 2:
                        L_fit = np.linspace(float(np.min(L_arr[mask])),
                                            float(np.max(L_arr[mask])), 200)
                        y_fit = _eval_powerlaw_fit_row(L_fit, final, bc_label, inverse=True)
                        m_fit = np.isfinite(y_fit) & (y_fit > 0.0)
                        if np.count_nonzero(m_fit):
                            ax_psi.plot(
                                L_fit[m_fit],
                                y_fit[m_fit],
                                color=color,
                                lw=1.5,
                                ls="-",
                                label="_nolegend_",
                            )
                            plotted_psi = True
                if plotted_psi:
                    psi_handles.append(
                        Line2D([], [], color=color, marker=marker, ls="-",
                               lw=1.5, markersize=5.5, label=bc_label)
                    )

            ax_beta.axhline(BETA / NU, color="tab:red", ls="--", lw=1.3,
                            label=rf"$\beta/\nu={BETA / NU:.4f}$")
            ax_beta.axvline(FINAL_LMIN_DEFAULT, color=cutoff_color, ls="--",
                            lw=1.35, alpha=0.68)
            ax_beta.set_xlabel(r"$L_{\min}$")
            ax_beta.set_ylabel(r"$\beta/\nu$")
            ax_beta.set_xticks(LMIN_SWEEP)
            ax_beta.tick_params(direction="in", which="both", top=True, right=True,
                                labelsize=TICK_FONTSIZE)
            apply_grid(ax_beta)
            beta_handles, beta_labels = ax_beta.get_legend_handles_labels()
            beta_handles.append(cutoff_handle)
            beta_labels.append(cutoff_handle.get_label())
            ax_beta.legend(handles=beta_handles, labels=beta_labels,
                           frameon=False, fontsize=LEGEND_FONTSIZE, loc="best")

            ax_psi.set_xscale("log")
            ax_psi.set_yscale("log")
            ax_psi.axvline(FINAL_LMIN_DEFAULT, color=cutoff_color, ls="--",
                           lw=1.35, alpha=0.68)
            ax_psi.set_xlabel(r"$L$")
            ax_psi.set_ylabel(ylabel)
            ax_psi.tick_params(direction="in", which="both", top=True, right=True,
                               labelsize=TICK_FONTSIZE)
            apply_grid(ax_psi)
            ax_psi.legend(handles=[*psi_handles, cutoff_handle],
                          frameon=False, fontsize=LEGEND_FONTSIZE, loc="best")

            save_psi_fig(fig2, filename)
            return

        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        for ax, bc_label in [(axes2[0], "PBC"), (axes2[1], "OBC")]:
            L_arr, Y_arr, sweep, final, beta_fit = _fit_context(bc_label)

            if L_arr.size:
                colors = colors_for_bc(bc_label)
                for L, Yv in zip(L_arr, Y_arr):
                    c = colors.get(L, "gray")
                    ax.plot(L, Yv, ls="None", marker="o", color=c)

            if final is not None and np.isfinite(final.get("A", np.nan)) and np.isfinite(beta_fit) and L_arr.size:
                mask = L_arr >= float(final["Lmin"])
                L_fit = np.linspace(float(np.min(L_arr[mask])),
                                    float(np.max(L_arr[mask])), 200)
                y_fit = _eval_powerlaw_fit_row(L_fit, final, bc_label, inverse=True)
                ax.plot(L_fit, y_fit,
                        color="black", lw=1.6,
                        label=rf"subleading $L_{{\min}}={final['Lmin']}$")

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(r"$L$")
            ax.set_ylabel(ylabel if ax is axes2[0] else "")
            ax.set_title(bc_label, loc="right")
            ax.tick_params(direction="in", which="both", top=True, right=True,
                           labelsize=TICK_FONTSIZE)
            apply_grid(ax)
            from matplotlib.lines import Line2D
            handles, labels = ax.get_legend_handles_labels()
            handles.append(Line2D([], [], color="none",
                                  label=rf"$\beta/\nu={beta_fit:.4f}$"))
            labels.append(rf"$\beta/\nu={beta_fit:.4f}$")
            if handles:
                ax.legend(handles=handles, labels=labels, frameon=False,
                          fontsize=LEGEND_FONTSIZE, loc="upper right", ncol=1)
        save_psi_fig(fig2, filename)

    _plot_psi_gc_fit("psiB", r"$\bar{\Psi}(g_c)$", "psi_bar_fss.pdf")
    _plot_psi_gc_fit("psiT", r"$\tilde{\Psi}(g_c)$", "psi_tilde_fss.pdf")


def print_hyperscaling_table(g_c_num, z_num, nu_num,
                             beta_over_nu, gamma_over_nu):
    beta_num = beta_over_nu * nu_num
    gamma_num = gamma_over_nu * nu_num
    alpha_num = 2.0 - nu_num * (1.0 + z_num)
    delta_num = 1.0 + gamma_num / beta_num if beta_num != 0.0 else np.nan
    rushbrooke = 2.0 * beta_num + gamma_num + alpha_num
    widom = beta_num * (delta_num - 1.0)
    widom_target = gamma_num
    beta_delta_plus_one = beta_num * (delta_num + 1.0)
    beta_delta_plus_one_target = 2.0 - alpha_num
    hyperscaling = nu_num * (1.0 + z_num)

    print("\n" + "=" * 60)
    print("  HYPERSCALING SUMMARY - Quantum Ising 1D")
    print("=" * 60)
    print(f"  {'Quantity':<32} {'Numerical':>10}  {'Target':>8}")
    print("-" * 60)
    rows = [
        ("g_c", g_c_num, G_C),
        ("z", z_num, Z_EXACT),
        ("nu", nu_num, NU),
        ("beta/nu", beta_over_nu, BETA / NU),
        ("gamma/nu", gamma_over_nu, GAMMA / NU),
        ("beta", beta_num, BETA),
        ("gamma", gamma_num, GAMMA),
        ("alpha", alpha_num, 0.0),
        ("delta", delta_num, 15.0),
        ("omega_PBC (fixed)", OMEGA_PBC, OMEGA_PBC),
        ("omega_OBC (fixed)", OMEGA_OBC, OMEGA_OBC),
        ("Rushbrooke 2beta+gamma+alpha", rushbrooke, 2.0),
        ("Widom beta(delta-1) = gamma", widom, widom_target),
        ("beta(delta+1) = 2 - alpha", beta_delta_plus_one, beta_delta_plus_one_target),
        ("Hyperscaling nu(d+z)", hyperscaling, 2.0),
    ]
    for name, num, target in rows:
        flag = "OK" if np.isfinite(num) and abs(num - target) < 0.05 else "NO"
        print(f"  {name:<32} {num:>10.4f}  {target:>8.4f}  {flag}")

    print("=" * 60 + "\n")


def print_approach_summary(primary_pbc: dict, primary_obc: dict) -> None:
    """Print the compact end-of-run summary for public and diagnostic approaches."""
    def _line(primary: dict, approach: str) -> str:
        bc = primary.get("bc", "")
        out_approach = public_approach_label(bc, approach)
        data = primary.get("channels", {}).get(approach, {})
        derived = primary.get("derived_exponents", {}).get(approach, {}).get("final", {})
        nu_inv = data.get("nu_inv", {}).get("value", np.nan)
        z_val = data.get("z_dynamic", {}).get("value", np.nan)
        beta_val = data.get("beta_over_nu", {}).get("value", np.nan)
        gamma_val = derived.get("gamma_over_nu", np.nan)
        return (
            f"  {bc:<3} {out_approach:<28} "
            f"{data.get('g_pc', np.nan):>11.8f} "
            f"{data.get('gpc_crossing_scale', ''):<10} "
            f"{nu_inv:>10.6f} {z_val:>10.6f} {beta_val:>13.6f} "
            f"{gamma_val:>13.6f} {derived.get('alpha', np.nan):>10.6f} "
            f"{derived.get('delta', np.nan):>10.6f}"
        )

    print("\nAPPROACH SUMMARY")
    print("  BC  approach                         g_pc scale          nu_inv          z  beta_over_nu gamma_over_nu      alpha      delta")
    print("  Main approaches")
    for primary, approaches in [
        (primary_pbc, ("leading", "raw_subleading")),
        (primary_obc, ("leading", "mixed_subleading")),
    ]:
        if not primary:
            continue
        for approach in approaches:
            if approach in primary.get("channels", {}):
                print(_line(primary, approach))
    print("  Diagnostic-only approaches")
    for primary, approaches in [(primary_obc, ("raw_subleading_diagnostic",))]:
        if not primary:
            continue
        for approach in approaches:
            if approach in primary.get("channels", {}):
                print(_line(primary, approach))
    print()


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------
def write_results(
    gcest,
    zpbc, zobc,
    omegapbc_gap, omegaobc_gap,
    nupbc, nuobc,
    betaovnupbc, betaovnuobc,
    gammaovnupbc, gammaovnuobc,
    Llist, gapdata,
    gapdata_obc=None,
    kink_results=None,
    betabarovnupbc=np.nan, betabarovnuobc=np.nan,
    primarypbc=None, primaryobc=None,
) -> None:
    """Write the deterministic FSS summary used by downstream plots."""
    from datetime import datetime

    path = FSS_DIR / "fss_results.txt"
    primarypbc = primarypbc or {}
    primaryobc = primaryobc or {}
    kink_results = kink_results or {}

    def fmt(v, decimals=6):
        return f"{float(v):.{decimals}f}" if np.isfinite(v) else "nan"

    def flag(vpbc, vobc, exact):
        if not np.isfinite(exact):
            return "n/a"
        vals = [v for v in (vpbc, vobc) if np.isfinite(v)]
        if not vals:
            return "---"
        return "OK" if all(abs(v - exact) < 0.05 for v in vals) else "NO"

    def block(primary: dict, key: str) -> dict:
        return primary.get(key, {}) if primary else {}

    def value(primary: dict, key: str) -> float:
        return block(primary, key).get("value", np.nan)

    def lmin_drift(primary: dict, key: str) -> float:
        return block(primary, key).get("lmin_drift", np.nan)

    def approach_data(primary: dict, approach: str) -> dict:
        return primary.get("channels", {}).get(approach, {}) if primary else {}

    def approach_gpc(primary: dict, approach: str) -> float:
        return approach_data(primary, approach).get("g_pc", np.nan)

    def derived(primary: dict, approach: str) -> dict:
        return primary.get("derived_exponents", {}).get(approach, {}) if primary else {}

    def derived_final(primary: dict, approach: str) -> dict:
        return derived(primary, approach).get("final", {})

    def derived_quantity(primary: dict, approach: str, quantity: str) -> float:
        return derived_final(primary, approach).get(quantity, np.nan)

    def derived_lmin_drift(primary: dict, approach: str, quantity: str) -> float:
        return derived(primary, approach).get(quantity, {}).get("lmin_drift", np.nan)

    def nu_from_inv(inv_nu: float) -> float:
        return 1.0 / inv_nu if np.isfinite(inv_nu) and inv_nu != 0.0 else np.nan

    def chiperp_log_final_row(primary: dict, approach: str, fit_kind: str) -> dict:
        log_block = approach_data(primary, approach).get("chiperp_log", {})
        return _chiperp_log_row_for_lmin(
            log_block,
            fit_kind,
            primary.get("final_lmin", FINAL_LMIN_DEFAULT),
        )

    inv_nu_pbc = value(primarypbc, "nu_inv_final")
    inv_nu_obc = value(primaryobc, "nu_inv_final")
    nu_pbc = nu_from_inv(inv_nu_pbc)
    nu_obc = nu_from_inv(inv_nu_obc)
    z_pbc = value(primarypbc, "z_dynamic_final")
    z_obc = value(primaryobc, "z_dynamic_final")
    beta_pbc = value(primarypbc, "beta_over_nu_final")
    beta_obc = value(primaryobc, "beta_over_nu_final")
    beta_bar_pbc = value(primarypbc, "beta_over_nu_bar_final")
    beta_bar_obc = value(primaryobc, "beta_over_nu_bar_final")
    gamma_pbc = gamma_over_nu_hyperscaling(beta_pbc, z=z_pbc)["value"]
    gamma_obc = gamma_over_nu_hyperscaling(beta_obc, z=z_obc)["value"]
    gamma_pbc_sweep = gamma_over_nu_sweep_from_hyperscaling(primarypbc)
    gamma_obc_sweep = gamma_over_nu_sweep_from_hyperscaling(primaryobc)
    gamma_pbc_drift = _lmin_drift(gamma_pbc_sweep, primarypbc.get("final_lmin", FINAL_LMIN_DEFAULT))
    gamma_obc_drift = _lmin_drift(gamma_obc_sweep, primaryobc.get("final_lmin", FINAL_LMIN_DEFAULT))
    pbc_final_approach = block(primarypbc, "nu_inv_final").get("gpc_approach", "raw_subleading")
    obc_final_approach = block(primaryobc, "nu_inv_final").get("gpc_approach", "mixed_subleading")

    best_nu = derived_quantity(primarypbc, pbc_final_approach, "nu")
    if not np.isfinite(best_nu):
        best_nu = nu_pbc
    best_z = z_pbc
    best_beta_over_nu = beta_pbc
    best_gamma_over_nu = gamma_pbc
    beta_num = derived_quantity(primarypbc, pbc_final_approach, "beta")
    gamma_num = derived_quantity(primarypbc, pbc_final_approach, "gamma")
    alpha_num = derived_quantity(primarypbc, pbc_final_approach, "alpha")
    delta_num = derived_quantity(primarypbc, pbc_final_approach, "delta")
    rushbrooke = (
        2.0 * beta_num + gamma_num + alpha_num
        if all(np.isfinite(x) for x in (beta_num, gamma_num, alpha_num))
        else np.nan
    )
    widom = beta_num * (delta_num - 1.0) if np.isfinite(beta_num) and np.isfinite(delta_num) else np.nan
    widom_target = gamma_num
    beta_delta_plus_one = (
        beta_num * (delta_num + 1.0)
        if np.isfinite(beta_num) and np.isfinite(delta_num)
        else np.nan
    )
    beta_delta_plus_one_target = 2.0 - alpha_num if np.isfinite(alpha_num) else np.nan
    hyperscal = best_nu * (1.0 + best_z) if np.isfinite(best_nu) and np.isfinite(best_z) else np.nan
    c_pbc = kink_results.get("PBC", {}).get("c", np.nan)
    c_obc = kink_results.get("OBC", {}).get("c", np.nan)

    table_rows = [
        ("gc final approach", approach_gpc(primarypbc, pbc_final_approach), approach_gpc(primaryobc, obc_final_approach), G_C, np.nan, np.nan),
        ("1/nu", inv_nu_pbc, inv_nu_obc, 1.0 / NU, lmin_drift(primarypbc, "nu_inv_final"), lmin_drift(primaryobc, "nu_inv_final")),
        ("nu", nu_pbc, nu_obc, NU, np.nan, np.nan),
        ("z", z_pbc, z_obc, Z_EXACT, lmin_drift(primarypbc, "z_dynamic_final"), lmin_drift(primaryobc, "z_dynamic_final")),
        ("beta/nu psi_tilde", beta_pbc, beta_obc, BETA / NU, lmin_drift(primarypbc, "beta_over_nu_final"), lmin_drift(primaryobc, "beta_over_nu_final")),
        ("beta/nu psi_bar", beta_bar_pbc, beta_bar_obc, BETA / NU, lmin_drift(primarypbc, "beta_over_nu_bar_final"), lmin_drift(primaryobc, "beta_over_nu_bar_final")),
        ("gamma/nu hyperscaling", gamma_pbc, gamma_obc, GAMMA / NU, gamma_pbc_drift, gamma_obc_drift),
        ("beta derived", beta_num, derived_quantity(primaryobc, obc_final_approach, "beta"), BETA, np.nan, np.nan),
        ("gamma derived", gamma_num, derived_quantity(primaryobc, obc_final_approach, "gamma"), GAMMA, np.nan, np.nan),
        ("alpha derived", alpha_num, derived_quantity(primaryobc, obc_final_approach, "alpha"), 0.0, derived_lmin_drift(primarypbc, pbc_final_approach, "alpha"), derived_lmin_drift(primaryobc, obc_final_approach, "alpha")),
        ("alpha chi_perp log", 0.0, 0.0, 0.0, 0.0, 0.0),
        ("delta derived", delta_num, derived_quantity(primaryobc, obc_final_approach, "delta"), 15.0, derived_lmin_drift(primarypbc, pbc_final_approach, "delta"), derived_lmin_drift(primaryobc, obc_final_approach, "delta")),
        ("c kink", c_pbc, c_obc, C_EXACT, np.nan, np.nan),
        ("Rushbrooke", rushbrooke, np.nan, 2.0, np.nan, np.nan),
        ("Widom beta(delta-1) = gamma", widom, np.nan, widom_target, np.nan, np.nan),
        ("beta(delta+1) = 2 - alpha", beta_delta_plus_one, np.nan, beta_delta_plus_one_target, np.nan, np.nan),
        ("Hyperscaling", hyperscal, np.nan, 2.0, np.nan, np.nan),
    ]

    with open(path, "w", encoding="utf-8") as f:
        def w(line=""):
            f.write(line + "\n")

        sep = "=" * 72
        w(sep)
        w("FSS RESULTS - 1D Quantum Ising Chain")
        w(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        w(f"System sizes used: {sorted(Llist)}")
        w("Final PBC estimates use the subleading approach.")
        w("OBC values are reported as boundary-condition diagnostics using the mixed subleading approach.")
        w(sep)
        w()

        w("Critical point diagnostics")
        w(f"  gap-L crossing PBC raw weighted: {fmt(gcest, 8)}   exact {G_C:.8f}")
        w()

        w("Approach pseudo-critical points")
        for primary in (primarypbc, primaryobc):
            if not primary:
                continue
            for approach, ch in primary.get("channels", {}).items():
                if primary["bc"] == "OBC" and approach == "raw_subleading_diagnostic":
                    continue
                out_approach = public_approach_label(primary["bc"], approach)
                w(
                    f"  {primary['bc']:<3} {out_approach:<28} "
                    f"g_pc={fmt(ch.get('g_pc', np.nan), 8)}  "
                    f"source={ch.get('gpc_source', '')}  "
                    f"scale={ch.get('gpc_crossing_scale', '')}"
                )
        w()

        w("Definition of OBC mixed subleading approach")
        w("  For spectral quantities we use Le=L+1/2 and residual omega=2.")
        w("  For order-parameter pseudo-observables we evaluate at g_pc obtained from Le crossings")
        w("  but fit the size dependence using raw L with omega=1.")
        w()

        w("Derived exponents by approach")
        w("  alpha_hyperscaling = 2 - nu(1+z) and delta are derived from scaling relations, not independently fitted.")
        w(f"  {'BC':<3} {'approach':<28} {'g_pc':>12} {'nu':>11} {'beta':>11} {'gamma':>11} {'alpha':>11} {'delta':>11} {'alpha drift':>13} {'delta drift':>13}")
        for primary in (primarypbc, primaryobc):
            if not primary:
                continue
            approaches = ("leading", "raw_subleading") if primary["bc"] == "PBC" else ("leading", "mixed_subleading")
            for approach in approaches:
                info = derived(primary, approach)
                if not info:
                    continue
                out_approach = public_approach_label(primary["bc"], approach)
                final = info.get("final", {})
                w(
                    f"  {primary['bc']:<3} {out_approach:<28} "
                    f"{fmt(info.get('g_pc', np.nan), 8):>12} "
                    f"{fmt(final.get('nu', np.nan), 6):>11} "
                    f"{fmt(final.get('beta', np.nan), 6):>11} "
                    f"{fmt(final.get('gamma', np.nan), 6):>11} "
                    f"{fmt(final.get('alpha', np.nan), 6):>11} "
                    f"{fmt(final.get('delta', np.nan), 6):>11} "
                    f"{fmt(info.get('alpha', {}).get('lmin_drift', np.nan), 6):>13} "
                    f"{fmt(info.get('delta', {}).get('lmin_drift', np.nan), 6):>13}"
                )
        w()

        w("Transverse susceptibility logarithmic alpha diagnostic")
        w("  C_x(L)=g_pc chi_x(L,g_pc).")
        w("  Fits: a0+a1 log L and a0+a1 log L+b L^(-omega).")
        w("  Because the singularity is logarithmic, this channel gives alpha_chiperp_log = 0.")
        w("  This is not a free power-law exponent fit.")
        w(f"  {'BC':<3} {'approach':<28} {'g_pc':>12} {'a1_leading':>14} {'a1_subleading':>17} {'alpha_chiperp_log':>19}")
        for primary in (primarypbc, primaryobc):
            if not primary:
                continue
            bc = primary.get("bc", "")
            for approach in _main_chiperp_approaches(primary):
                lead = chiperp_log_final_row(primary, approach, "leading")
                sub = chiperp_log_final_row(primary, approach, "subleading")
                out_approach = public_approach_label(bc, approach)
                w(
                    f"  {bc:<3} {out_approach:<28} "
                    f"{fmt(approach_gpc(primary, approach), 8):>12} "
                    f"{fmt(lead.get('a1', np.nan), 6):>14} "
                    f"{fmt(sub.get('a1', np.nan), 6):>17} "
                    f"{fmt(0.0, 6):>19}"
                )
        w()

        w("Master summary")
        w(f"  {'Quantity':<28} {'Final PBC value':>16}  {'OBC diagnostic':>16}  {'exact/target':>12}  {'flag':>4}  {'PBC lmin_drift':>16}  {'OBC lmin_drift':>16}")
        for label, vp, vo, exact, dp, do in table_rows:
            exact_str = f"{exact:.6f}" if np.isfinite(exact) else "---"
            w(
                f"  {label:<28} {fmt(vp, 6):>16}  {fmt(vo, 6):>16}  "
                f"{exact_str:>12}  {flag(vp, vo, exact):>4}  "
                f"{fmt(dp, 6):>16}  {fmt(do, 6):>16}"
            )
        w()

        w("Scaling relations from final PBC values")
        w(f"  Rushbrooke 2 beta + gamma + alpha: {fmt(rushbrooke, 6)}   exact 2.000000")
        w(f"  Widom beta(delta-1) = gamma:      {fmt(widom, 6)}   target {fmt(widom_target, 6)}")
        w(f"  beta(delta+1) = 2 - alpha:        {fmt(beta_delta_plus_one, 6)}   target {fmt(beta_delta_plus_one_target, 6)}")
        w(f"  Hyperscaling nu(1+z):             {fmt(hyperscal, 6)}   exact 2.000000")
        w()

        w("# MACHINE-READABLE KEY-VALUE PAIRS")
        w(f"gc_num = {fmt(gcest, 8)}")
        w(f"GPC_PBC_LEADING = {fmt(approach_gpc(primarypbc, 'leading'), 8)}")
        w(f"GPC_PBC_SUBLEADING = {fmt(approach_gpc(primarypbc, 'raw_subleading'), 8)}")
        w(f"GPC_OBC_LEADING = {fmt(approach_gpc(primaryobc, 'leading'), 8)}")
        w(f"GPC_OBC_RAW_SUBLEADING_DIAGNOSTIC = {fmt(approach_gpc(primaryobc, 'raw_subleading_diagnostic'), 8)}")
        w(f"GPC_OBC_MIXED_SUBLEADING = {fmt(approach_gpc(primaryobc, 'mixed_subleading'), 8)}")
        w("GPC_SOURCE_PBC_LEADING = raw_L_leading")
        w("GPC_SOURCE_PBC_SUBLEADING = raw_L_subleading")
        w("GPC_SOURCE_OBC_LEADING = raw_L_leading")
        w("GPC_SOURCE_OBC_RAW_SUBLEADING_DIAGNOSTIC = raw_L_subleading")
        w("GPC_SOURCE_OBC_MIXED_SUBLEADING = Le=L+1/2")
        w(f"nu_best = {fmt(best_nu, 8)}")
        w(f"nu_pbc = {fmt(nu_pbc, 8)}")
        w(f"nu_obc = {fmt(nu_obc, 8)}")
        w(f"beta_ov_nu_best = {fmt(best_beta_over_nu, 8)}")
        w(f"beta_ov_nu_pbc = {fmt(beta_pbc, 8)}")
        w(f"beta_ov_nu_obc = {fmt(beta_obc, 8)}")
        w(f"beta_ov_nu_bar_pbc = {fmt(beta_bar_pbc, 8)}")
        w(f"beta_ov_nu_bar_obc = {fmt(beta_bar_obc, 8)}")
        w(f"gamma_ov_nu_best = {fmt(best_gamma_over_nu, 8)}")
        w(f"gamma_ov_nu_pbc = {fmt(gamma_pbc, 8)}")
        w(f"gamma_ov_nu_obc = {fmt(gamma_obc, 8)}")
        w(f"z_best = {fmt(best_z, 8)}")
        w(f"z_pbc = {fmt(z_pbc, 8)}")
        w(f"z_obc = {fmt(z_obc, 8)}")
        w(f"c_pbc = {fmt(c_pbc, 8)}")
        w(f"c_obc = {fmt(c_obc, 8)}")
        for name, primary, key in [
            ("INV_NU_PBC_LEADING", primarypbc, "nu_inv_leading"),
            ("INV_NU_PBC_SUBLEADING", primarypbc, "nu_inv_raw_subleading"),
            ("INV_NU_OBC_LEADING", primaryobc, "nu_inv_leading"),
            ("INV_NU_OBC_MIXED_SUBLEADING", primaryobc, "nu_inv_mixed_subleading"),
            ("Z_PBC_LEADING", primarypbc, "z_dynamic_leading"),
            ("Z_PBC_SUBLEADING", primarypbc, "z_dynamic_raw_subleading"),
            ("Z_OBC_LEADING", primaryobc, "z_dynamic_leading"),
            ("Z_OBC_MIXED_SUBLEADING", primaryobc, "z_dynamic_mixed_subleading"),
            ("BETA_OVER_NU_PBC_PSIT_LEADING", primarypbc, "beta_over_nu_leading"),
            ("BETA_OVER_NU_PBC_PSIT_SUBLEADING", primarypbc, "beta_over_nu_raw_subleading"),
            ("BETA_OVER_NU_PBC_PSIB_LEADING", primarypbc, "beta_over_nu_bar_leading"),
            ("BETA_OVER_NU_PBC_PSIB_SUBLEADING", primarypbc, "beta_over_nu_bar_raw_subleading"),
            ("BETA_OVER_NU_OBC_PSIT_LEADING", primaryobc, "beta_over_nu_leading"),
            ("BETA_OVER_NU_OBC_PSIT_MIXED_SUBLEADING", primaryobc, "beta_over_nu_mixed_subleading"),
            ("BETA_OVER_NU_OBC_PSIB_LEADING", primaryobc, "beta_over_nu_bar_leading"),
            ("BETA_OVER_NU_OBC_PSIB_MIXED_SUBLEADING", primaryobc, "beta_over_nu_bar_mixed_subleading"),
        ]:
            w(f"{name} = {fmt(value(primary, key), 8)}")
            w(f"{name}_LMIN_DRIFT = {fmt(lmin_drift(primary, key), 8)}")
        w(f"GAMMA_OVER_NU_PBC_FINAL = {fmt(gamma_pbc, 8)}")
        w(f"GAMMA_OVER_NU_PBC_FINAL_LMIN_DRIFT = {fmt(gamma_pbc_drift, 8)}")
        w(f"GAMMA_OVER_NU_OBC_DIAGNOSTIC = {fmt(gamma_obc, 8)}")
        w(f"GAMMA_OVER_NU_OBC_DIAGNOSTIC_LMIN_DRIFT = {fmt(gamma_obc_drift, 8)}")
        for primary in (primarypbc, primaryobc):
            if not primary:
                continue
            bc_key = primary.get("bc", "").upper()
            for approach, info in primary.get("derived_exponents", {}).items():
                if bc_key == "OBC" and approach == "raw_subleading_diagnostic":
                    continue
                prefix = f"DERIVED_{bc_key}_{approach.upper()}"
                if bc_key == "PBC" and approach == "raw_subleading":
                    prefix = "DERIVED_PBC_SUBLEADING"
                final = info.get("final", {})
                for name in ("nu", "beta", "gamma", "alpha", "delta"):
                    w(f"{prefix}_{name.upper()} = {fmt(final.get(name, np.nan), 8)}")
                w(f"{prefix}_ALPHA_LMIN_DRIFT = {fmt(info.get('alpha', {}).get('lmin_drift', np.nan), 8)}")
                w(f"{prefix}_DELTA_LMIN_DRIFT = {fmt(info.get('delta', {}).get('lmin_drift', np.nan), 8)}")
        for primary in (primarypbc, primaryobc):
            if not primary:
                continue
            bc_key = primary.get("bc", "").upper()
            for approach in _main_chiperp_approaches(primary):
                log_block = approach_data(primary, approach).get("chiperp_log", {})
                out_approach = public_approach_label(bc_key, approach).upper()
                suffix = out_approach.replace(" ", "_").replace("=", "").replace("+", "")
                prefix = f"{bc_key}_{suffix}"
                lead = log_block.get("final_leading", {})
                sub = log_block.get("final_subleading", {})
                w(f"ALPHA_CHIPERP_LOG_{prefix} = {fmt(0.0, 8)}")
                w(f"CHIPERP_LOG_SLOPE_{prefix}_LEADING = {fmt(lead.get('a1', np.nan), 8)}")
                w(f"CHIPERP_LOG_SLOPE_{prefix}_SUBLEADING = {fmt(sub.get('a1', np.nan), 8)}")
                w(f"CHIPERP_LOG_RESID_{prefix}_LEADING = {fmt(lead.get('resid_rms', np.nan), 8)}")
                w(f"CHIPERP_LOG_RESID_{prefix}_SUBLEADING = {fmt(sub.get('resid_rms', np.nan), 8)}")
        w(sep)

    print(f"[OK] {path.relative_to(PROJECT_ROOT)}")


def warn_forbidden_results_text(path: Path = FSS_DIR / "fss_results.txt") -> None:
    """Warn if stale public labels remain in the human-readable summary."""
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    forbidden = [
        "GPC_OBC_RAW_SUBLEADING_USED",
        "GPC_SOURCE_OBC_RAW_SUBLEADING_USED",
        "Channel pseudo-critical points",
        "Derived exponents by channel",
    ]
    for token in forbidden:
        if token in text:
            warnings.warn(f"Stale results label found in {path.name}: {token}", RuntimeWarning)


def remove_legacy_chiperp_log_outputs() -> None:
    """Remove stale outputs from retired chi_perp power-law pipelines."""
    for path in (
        FSS_DIR / "alpha_chix_power_sweep.dat",
        FSS_DIR / "alpha_chiperp_power_sweep.dat",
        PLOT_DIR / "alpha_chix_power_sweep.pdf",
        PLOT_DIR / "alpha_chix_power_fit.pdf",
        PLOT_DIR / "alpha_chiperp_power_sweep.pdf",
        PLOT_DIR / "alpha_chiperp_power_fit.pdf",
    ):
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            warnings.warn(f"Could not remove stale output {path.name}: {exc}", RuntimeWarning)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="FSS analysis for 1D quantum Ising chain.")
    parser.add_argument("--lanczos", action="store_true", help="Use Lanczos gap files (gap_lz_L*.dat)")
    parser.add_argument("--L", nargs="+", type=int, default=ALL_SIZES,
                        help="System sizes (default: all configured sizes)")
    parser.add_argument("--L-large", nargs="+", type=int, default=LZ_SIZES,
                        help="Additional sizes from Lanczos")
    parser.add_argument("--chizfd-only", action="store_true",
                        help="Only plot diagnostic chi_z finite-difference data")
    parser.add_argument("--chizfd-dh-tag", default="dh_5e-04",
                        help="Subdirectory under data/h_null/chiz_fd for chi_z FD data")
    parser.add_argument("--allow-partial-chizfd", action="store_true",
                        help="Plot partial chi_z FD files instead of skipping them")
    parser.add_argument(
        "--full-diagnostics",
        action="store_true",
        help="Also generate secondary diagnostic plots.",
    )
    args = parser.parse_args()

    L_base = sorted(set(args.L) | set(ALL_SIZES))
    if args.lanczos:
        L_base = sorted(set(L_base + args.L_large))
    L_list_pbc = sorted(set(L_base) | set(PBC_EXTRA_SIZES))
    L_list_obc = L_base

    sns.set_theme(style="ticks", context="paper", font_scale=FONT_SCALE)
    plt.rcParams.update({
        "axes.spines.right": True,
        "axes.spines.top": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "font.size": TEXT_FONTSIZE,
        "axes.labelsize": AXIS_LABEL_FONTSIZE,
        "axes.titlesize": TITLE_FONTSIZE,
        "xtick.labelsize": TICK_FONTSIZE,
        "ytick.labelsize": TICK_FONTSIZE,
        "legend.fontsize": LEGEND_FONTSIZE,
    })

    if args.chizfd_only:
        print("[fss_h_null_analysis.py]  Loading chi_z finite-difference data ...")
        chiz_pbc = load_chiz_fd_data(
            dh_tag=args.chizfd_dh_tag,
            sizes=L_list_pbc,
            bc="pbc",
            allow_partial=args.allow_partial_chizfd,
        )
        chiz_obc = load_chiz_fd_data(
            dh_tag=args.chizfd_dh_tag,
            sizes=L_list_obc,
            bc="obc",
            allow_partial=args.allow_partial_chizfd,
        )
        plot_chiz_fd_fss_panels(chiz_pbc, chiz_obc)
        print("Done chiz_fd FSS panels")
        return

    print("[fss_h_null_analysis.py]  Loading gap data ...")
    gap_data = gather_gap_data(L_list_pbc, pbc=True, lanczos=args.lanczos)
    gap_data_obc = gather_gap_data(L_list_obc, pbc=False, lanczos=args.lanczos)
    if not gap_data:
        print("[ERROR] No gap data found. Run ising_static (or ising_lanczos) first.")
        return

    data_pbc = {L: (gd, None, "") for L, gd in gap_data.items()}
    data_obc = {L: (gd, None, "") for L, gd in gap_data_obc.items()}

    gap_cross_pbc = gap_scaled_crossings_table(gap_data, bc="pbc", use_effective_length=False)
    gap_cross_obc = gap_scaled_crossings_table(gap_data_obc, bc="obc", use_effective_length=False)
    gap_cross_obc_le = (
        gap_scaled_crossings_table(gap_data_obc, bc="obc", use_effective_length=True)
        if gap_data_obc else []
    )
    write_gap_crossing_table(gap_cross_pbc, "PBC")
    if gap_data_obc:
        write_gap_crossing_table(gap_cross_obc + gap_cross_obc_le, "OBC")

    gpc_specs_pbc = make_gpc_specs("PBC", gap_cross_pbc)
    gpc_specs_obc = make_gpc_specs("OBC", gap_cross_obc, gap_cross_obc_le) if gap_data_obc else {}

    gc_est = gpc_specs_pbc.get("raw_subleading", {}).get("value", np.nan)
    if not np.isfinite(gc_est):
        gc_est = estimate_gc_from_crossing(gap_data)
    print(f"\n  g_c estimate from deterministic gap*L crossings: {gc_est:.6f}  (exact: 1.000000)")

    print("\n[fss_h_null_analysis.py]  Generating plots -> plots/h_null/fss/")
    suffix = "Lanczos" if args.lanczos else "full ED"
    gc_est_obc = gpc_specs_obc.get("mixed_subleading", {}).get("value", np.nan)
    gc_est_obc_diagnostic = gc_est_obc
    if gap_data_obc and not np.isfinite(gc_est_obc):
        warnings.warn(
            "OBC mixed_subleading g_pc is not finite; using exact g_c only for diagnostic plots.",
            RuntimeWarning,
        )
        gc_est_obc_diagnostic = G_C
    plot_gap_scaled_crossing(
        gap_data,
        gap_data_obc=gap_data_obc,
        title_suffix=suffix,
        g_pc=gc_est,
        g_pc_obc=gc_est_obc,
    )

    if args.full_diagnostics:
        gapL_fit_results = plot_gapL_subleading(
            data_pbc,
            data_obc,
            g_pc_pbc=gc_est,
            g_pc_obc=gc_est_obc_diagnostic,
        )
        for bc_label, res in gapL_fit_results.items():
            if not res:
                continue
            print(f"  gapL [{bc_label}]:  C={res['C']:.6f}"
                  f"  B={res['B']:.4f}  omega_fixed={res['omega']:.1f}"
                  "  (OLS subleading diagnostic)")
        plot_delta0_vs_L(data_pbc, data_obc)

    kink_results = plot_kink_velocity(
        data_pbc,
        data_obc,
        g_pc_pbc=gc_est,
        g_pc_obc=gc_est_obc,
    )

    obs_pbc = load_obs_data(L_list_pbc, "pbc")
    obs_obc = load_obs_data(L_list_obc, "obc")

    g_c_num = gc_est
    g_c_num_obc = gpc_specs_obc.get("mixed_subleading", {}).get("value", g_c_num)
    primary_pbc = build_primary_fss(gap_data, obs_pbc, "PBC", gap_cross_pbc, gpc_specs_pbc)
    primary_obc = (
        build_primary_fss(gap_data_obc, obs_obc, "OBC", gap_cross_obc, gpc_specs_obc)
        if gap_data_obc else {}
    )
    remove_legacy_chiperp_log_outputs()
    attach_chiperp_log_diagnostics(primary_pbc, obs_pbc)
    if primary_obc:
        attach_chiperp_log_diagnostics(primary_obc, obs_obc)
    write_primary_tables(primary_pbc)
    if primary_obc:
        write_primary_tables(primary_obc)
    write_nu_inv_sweep(primary_pbc, primary_obc)
    write_beta_over_nu_sweep(primary_pbc, primary_obc)
    write_exponent_sweeps(primary_pbc, primary_obc)
    write_derived_exponents_sweep(primary_pbc, primary_obc)
    write_z_dynamic_sweep(primary_pbc, primary_obc)
    write_alpha_chiperp_sweep(primary_pbc, primary_obc)
    plot_primary_robustness(primary_pbc, primary_obc)
    plot_alpha_chiperp_logfit(primary_pbc, primary_obc)
    if args.full_diagnostics:
        plot_gap_derivative_nu(primary_pbc, primary_obc)
    nu_inv_pbc = primary_pbc.get("nu_inv_final", {}).get("value", np.nan)
    nu_inv_obc = primary_obc.get("nu_inv_final", {}).get("value", np.nan) if primary_obc else np.nan
    nu_num_pbc = 1.0 / nu_inv_pbc if np.isfinite(nu_inv_pbc) and nu_inv_pbc != 0.0 else np.nan
    nu_num_obc = 1.0 / nu_inv_obc if np.isfinite(nu_inv_obc) and nu_inv_obc != 0.0 else np.nan
    nu_num = nu_num_pbc
    if not np.isfinite(nu_num) or nu_num <= 0.0:
        nu_num = NU
    z_report_pbc = primary_pbc.get("z_dynamic_final", {}).get("value", np.nan)
    z_report_obc = primary_obc.get("z_dynamic_final", {}).get("value", np.nan) if primary_obc else np.nan
    z_num = z_report_pbc if np.isfinite(z_report_pbc) else Z_EXACT

    beta_ov_nu_pbc = primary_pbc.get("beta_over_nu_final", {}).get("value", np.nan)
    beta_ov_nu_obc = primary_obc.get("beta_over_nu_final", {}).get("value", np.nan) if primary_obc else np.nan
    beta_bar_ov_nu_pbc = primary_pbc.get("beta_over_nu_bar_final", {}).get("value", np.nan)
    beta_bar_ov_nu_obc = primary_obc.get("beta_over_nu_bar_final", {}).get("value", np.nan) if primary_obc else np.nan
    hs_pbc = gamma_over_nu_hyperscaling(beta_ov_nu_pbc, z=z_report_pbc)
    hs_obc = gamma_over_nu_hyperscaling(beta_ov_nu_obc, z=z_report_obc)
    gamma_ov_nu_pbc = hs_pbc["value"]
    gamma_ov_nu_obc = hs_obc["value"]

    beta_ov_nu = beta_ov_nu_pbc
    gamma_ov_nu = gamma_ov_nu_pbc
    if not np.isfinite(beta_ov_nu):
        beta_ov_nu = BETA / NU
    if not np.isfinite(gamma_ov_nu):
        gamma_ov_nu = GAMMA / NU

    plot_chiperp_fss_panels(
        obs_pbc,
        obs_obc,
        g_c_num=g_c_num,
        g_c_num_obc=g_c_num_obc,
        nu_num=nu_num,
        nu_num_obc=nu_num_obc,
    )

    if args.full_diagnostics:
        plot_fss_collapse(
            obs_pbc, obs_obc, g_c_num, nu_num, beta_ov_nu, gamma_ov_nu,
            g_c_num_obc=g_c_num_obc,
            nu_num_obc=nu_num_obc,
            beta_over_nu_obc=beta_ov_nu_obc,
        )
    plot_binder_collapse(
        obs_pbc, obs_obc, g_c_num, nu_num,
        g_c_num_obc=g_c_num_obc,
        nu_num_obc=nu_num_obc,
    )

    print("[fss_h_null_analysis.py]  Loading chi_z finite-difference data ...")
    chiz_pbc = load_chiz_fd_data(
        dh_tag=args.chizfd_dh_tag,
        sizes=L_list_pbc,
        bc="pbc",
        allow_partial=args.allow_partial_chizfd,
    )
    chiz_obc = load_chiz_fd_data(
        dh_tag=args.chizfd_dh_tag,
        sizes=L_list_obc,
        bc="obc",
        allow_partial=args.allow_partial_chizfd,
    )
    if chiz_pbc or chiz_obc:
        plot_chiz_fd_fss_panels(
            chiz_pbc,
            chiz_obc,
            g_c_num=g_c_num,
            g_c_num_obc=g_c_num_obc,
            nu_num=nu_num,
            nu_num_obc=nu_num_obc,
            gamma_over_nu=gamma_ov_nu,
            gamma_over_nu_obc=gamma_ov_nu_obc,
        )
        plot_fss_all_temporary(
            obs_pbc,
            obs_obc,
            chiz_pbc,
            chiz_obc,
            g_c_num,
            nu_num,
            beta_ov_nu,
            gamma_ov_nu,
            g_c_num_obc=g_c_num_obc,
            nu_num_obc=nu_num_obc,
            beta_over_nu_obc=beta_ov_nu_obc,
            gamma_over_nu_obc=gamma_ov_nu_obc,
        )

    beta_for_psi = beta_ov_nu_pbc
    if not np.isfinite(beta_for_psi):
        beta_for_psi = BETA / NU
    plot_psi_fss(obs_pbc, obs_obc, g_c_num, nu_num, beta_for_psi,
                 primary_pbc=primary_pbc, primary_obc=primary_obc)

    print_hyperscaling_table(g_c_num, z_num, nu_num, beta_ov_nu, gamma_ov_nu)
    print_approach_summary(primary_pbc, primary_obc)

    write_results(
        gc_est,
        z_report_pbc, z_report_obc,
        OMEGA_PBC, OMEGA_OBC,     # fixed theory values
        nu_num_pbc, nu_num_obc,
        beta_ov_nu_pbc, beta_ov_nu_obc,
        gamma_ov_nu_pbc, gamma_ov_nu_obc,
        sorted(set(gap_data.keys()) | set(gap_data_obc.keys())), gap_data,
        gapdata_obc=gap_data_obc,
        kink_results=kink_results,
        betabarovnupbc=beta_bar_ov_nu_pbc, betabarovnuobc=beta_bar_ov_nu_obc,
        primarypbc=primary_pbc,
        primaryobc=primary_obc,
    )
    warn_forbidden_results_text()

    print("\n[fss_h_null_analysis.py]  Done.")


if __name__ == "__main__":
    main()
