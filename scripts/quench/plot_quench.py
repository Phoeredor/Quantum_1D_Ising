#!/usr/bin/env python3
"""
Plot ED full-spectrum quench scaling data.

Expected inputs:
  data/quench/cqt/quench_cqt_<bc>_LXX.dat
  data/quench/foqt/quench_foqt_<bc>_g0.500_LXX.dat
  data/quench/foqt/quench_foqt_<bc>_g0.900_LXX.dat
  data/quench/loschmidt/loschmidt_<bc>_Phi1.000_LXX.dat
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "quench"
PLOT_DIR = PROJECT_ROOT / "plots" / "quench"
MPL_CACHE_DIR = Path("/tmp/qising_1d_matplotlib_cache/quench")
if "MPLCONFIGDIR" not in os.environ:
    MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(MPL_CACHE_DIR)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CQT_RE = re.compile(r"^quench_cqt_(pbc|obc)_L(\d+)\.dat$")
FOQT_RE = re.compile(r"^quench_foqt_(pbc|obc)_g([0-9]+\.[0-9]+)_L(\d+)\.dat$")
LOSCH_RE = re.compile(r"^loschmidt_(pbc|obc)_Phi([0-9]+\.[0-9]+)_L(\d+)\.dat$")

ALL_SIZES = [4, 6, 8, 10, 12, 14]
PBC_COLOR_VALUES = np.linspace(0.2, 0.95, len(ALL_SIZES))
OBC_COLOR_VALUES = np.linspace(0.2, 0.95, len(ALL_SIZES))
COLORS_PBC = {L: plt.cm.plasma(v) for L, v in zip(ALL_SIZES, PBC_COLOR_VALUES)}
COLORS_OBC = {L: plt.cm.viridis(v) for L, v in zip(ALL_SIZES, OBC_COLOR_VALUES)}
LEGEND_Y_ANCHOR = -0.08
FONT_SCALE = 1.6
AXIS_LABEL_FONTSIZE = 18
TICK_FONTSIZE = 15
LEGEND_FONTSIZE = 15
TITLE_FONTSIZE = 15
TEXT_FONTSIZE = 15
DATA_LINEWIDTH = 1.7
REFERENCE_LINEWIDTH = 1.5


@dataclass(frozen=True)
class DataFile:
    path: Path
    bc: str
    L: int
    g: float | None = None
    phi: float | None = None


def load_table(path: Path, min_cols: int) -> np.ndarray | None:
    try:
        arr = np.loadtxt(path, comments="#")
    except Exception as exc:
        print(f"[WARN] cannot read {path}: {exc}")
        return None
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] < min_cols:
        print(f"[WARN] malformed table {path}: shape={arr.shape}")
        return None
    return arr


def discover_cqt(data_dir: Path) -> list[DataFile]:
    out: list[DataFile] = []
    cqt_dir = data_dir / "cqt"
    for path in sorted(cqt_dir.glob("quench_cqt_*.dat")):
        match = CQT_RE.match(path.name)
        if match:
            out.append(DataFile(path=path, bc=match.group(1), L=int(match.group(2))))
    return out


def discover_foqt(data_dir: Path) -> list[DataFile]:
    out: list[DataFile] = []
    foqt_dir = data_dir / "foqt"
    for path in sorted(foqt_dir.glob("quench_foqt_*.dat")):
        match = FOQT_RE.match(path.name)
        if match:
            out.append(
                DataFile(
                    path=path,
                    bc=match.group(1),
                    g=float(match.group(2)),
                    L=int(match.group(3)),
                )
            )
    return out


def discover_loschmidt(data_dir: Path) -> list[DataFile]:
    out: list[DataFile] = []
    losch_dir = data_dir / "loschmidt"
    for path in sorted(losch_dir.glob("loschmidt_*.dat")):
        match = LOSCH_RE.match(path.name)
        if match:
            out.append(
                DataFile(
                    path=path,
                    bc=match.group(1),
                    L=int(match.group(3)),
                    phi=float(match.group(2)),
                )
            )
    return out


def label_for_bc(bc: str) -> str:
    return "PBC" if bc == "pbc" else "OBC"


def colors_for_bc(bc: str) -> dict[int, tuple[float, float, float, float]]:
    return COLORS_PBC if bc == "pbc" else COLORS_OBC


class DoubleBCLine:
    def __init__(self, L: int) -> None:
        self.L = int(L)


class DoubleBCLineHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        artists = [
            Line2D(
                [x0, x0 + width],
                [y0 + 0.68 * height, y0 + 0.68 * height],
                color=COLORS_PBC.get(orig_handle.L, "gray"),
                ls="-",
                lw=1.9,
                solid_capstyle="round",
            ),
            Line2D(
                [x0, x0 + width],
                [y0 + 0.32 * height, y0 + 0.32 * height],
                color=COLORS_OBC.get(orig_handle.L, "gray"),
                ls="-",
                lw=1.9,
                solid_capstyle="round",
            ),
        ]
        for artist in artists:
            artist.set_transform(handlebox.get_transform())
            handlebox.add_artist(artist)
        return artists[0]


def add_external_legend(
    fig: plt.Figure,
    L_values: list[int],
    *,
    extra_handles: list[Line2D] | None = None,
    extra_labels: list[str] | None = None,
) -> None:
    handles: list[object] = [DoubleBCLine(L) for L in ALL_SIZES if L in set(L_values)]
    labels = [rf"$L={L}$" for L in ALL_SIZES if L in set(L_values)]

    if extra_handles and extra_labels:
        handles.extend(extra_handles)
        labels.extend(extra_labels)

    if not handles:
        return

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, LEGEND_Y_ANCHOR),
        bbox_transform=fig.transFigure,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        ncol=6,
        columnspacing=1.7,
        handlelength=2.2,
        handletextpad=0.8,
        handler_map={DoubleBCLine: DoubleBCLineHandler()},
    )


def add_panel_tag(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.96,
        0.94,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12,
    )


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "font.size": TEXT_FONTSIZE,
            "axes.labelsize": AXIS_LABEL_FONTSIZE,
            "axes.titlesize": TITLE_FONTSIZE,
            "xtick.labelsize": TICK_FONTSIZE,
            "ytick.labelsize": TICK_FONTSIZE,
            "legend.fontsize": LEGEND_FONTSIZE,
            "path.simplify": False,
        }
    )


def plot_cqt(files: list[DataFile]) -> None:
    if not files:
        print("[WARN] no CQT data found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.5), sharey=True, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.03, wspace=0.02, hspace=0.02)
    L_values = sorted({f.L for f in files})
    any_data = False

    for ax, bc in zip(axes, ("pbc", "obc")):
        colors = colors_for_bc(bc)
        for f in sorted([item for item in files if item.bc == bc], key=lambda x: x.L):
            data = load_table(f.path, min_cols=13)
            if data is None:
                continue
            theta = data[:, 0]
            psi_scaled = data[:, 8]
            alpha = 0.85 if f.L in (4, 6) else 1.0
            ax.plot(
                theta,
                psi_scaled,
                "-",
                color=colors[f.L],
                lw=DATA_LINEWIDTH,
                alpha=alpha,
                solid_capstyle="round",
                solid_joinstyle="round",
                antialiased=True,
            )
            any_data = True

        ax.axhline(0.0, color="0.55", lw=REFERENCE_LINEWIDTH, ls=":")
        ax.set_xlabel(r"$\Theta=t/L^z$", fontsize=AXIS_LABEL_FONTSIZE)
        ax.grid(alpha=0.25, ls=":")
        ax.set_title(label_for_bc(bc), loc="right", fontsize=TITLE_FONTSIZE)
        ax.tick_params(labelsize=TICK_FONTSIZE, direction="in", which="both", top=True, right=True)

    axes[0].set_ylabel(r"$\Psi_{\rm scaled}=M_z L^{\beta/\nu}$", fontsize=AXIS_LABEL_FONTSIZE)
    axes[1].tick_params(labelleft=False)

    if any_data:
        add_external_legend(fig, L_values)
        out = PLOT_DIR / "cqt_dynamic_fss.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"[OK] wrote {out}")
    plt.close(fig)


def two_level_curve(theta: np.ndarray, kappa0: float, kappa1: float) -> np.ndarray:
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]])
    h0 = 0.5 * np.array([[kappa0, 1.0], [1.0, -kappa0]])
    h1 = 0.5 * np.array([[kappa1, 1.0], [1.0, -kappa1]])

    eval0, evec0 = np.linalg.eigh(h0)
    eval1, evec1 = np.linalg.eigh(h1)
    psi0 = evec0[:, int(np.argmin(eval0))]
    coeff = evec1.T @ psi0

    phase = np.exp(-1j * np.outer(theta, eval1))
    states = (phase * coeff) @ evec1.T
    return np.real(np.einsum("bi,ij,bj->b", np.conjugate(states), sigma_z, states))


def align_two_level_sign(theory: np.ndarray, data_curves: list[np.ndarray]) -> np.ndarray:
    if not data_curves:
        return -theory
    first_values = [curve[0] for curve in data_curves if curve.size and np.isfinite(curve[0])]
    if not first_values:
        return -theory
    data_sign = np.sign(np.mean(first_values))
    theory_sign = np.sign(theory[0])
    if data_sign != 0.0 and theory_sign != 0.0 and data_sign != theory_sign:
        # The requested H2 convention gives the opposite sigma_z orientation
        # from the lattice Mz convention for kappa0=+1. Only the plotted
        # two-level curve is flipped; numerical data are never modified.
        return -theory
    return theory


def inset_limits_near_theory_peak(
    theta_grid: np.ndarray,
    theory: np.ndarray,
    curves: list[tuple[np.ndarray, np.ndarray]],
    g_target: float,
    *,
    half_width: float | None = None,
    pad_frac: float = 0.10,
    min_yspan: float | None = None,
) -> tuple[float, float, float, float]:
    peak_mask = (theta_grid >= 3.0) & (theta_grid <= 6.0) & np.isfinite(theory)
    if not np.any(peak_mask):
        return 3.0, 6.0, -1.0, 1.0

    theta_window = theta_grid[peak_mask]
    theory_window = theory[peak_mask]
    theta_peak = float(theta_window[int(np.argmax(theory_window))])
    if half_width is None:
        half_width = 0.65 if g_target < 0.75 else 0.85
    xlo = theta_peak - half_width
    xhi = theta_peak + half_width

    y_parts: list[np.ndarray] = []
    theory_mask = (theta_grid >= xlo) & (theta_grid <= xhi) & np.isfinite(theory)
    if np.any(theory_mask):
        y_parts.append(theory[theory_mask])

    for x, y in curves:
        mask = (x >= xlo) & (x <= xhi) & np.isfinite(x) & np.isfinite(y)
        if np.any(mask):
            y_parts.append(y[mask])

    if not y_parts:
        return xlo, xhi, -1.0, 1.0

    y_all = np.concatenate(y_parts)
    ymin = float(np.min(y_all))
    ymax = float(np.max(y_all))
    yspan = ymax - ymin
    pad = pad_frac * yspan if yspan > 0.0 else 0.0
    ymin -= pad
    ymax += pad

    if min_yspan is None:
        min_yspan = 0.05 if g_target < 0.75 else 0.15
    if ymax - ymin < min_yspan:
        center = 0.5 * (ymin + ymax)
        ymin = center - 0.5 * min_yspan
        ymax = center + 0.5 * min_yspan

    return xlo, xhi, ymin, ymax


def plot_foqt(files: list[DataFile]) -> None:
    if not files:
        print("[WARN] no FOQT data found")
        return

    g_targets = [0.5, 0.9]
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14.0, 10.0),
        sharex="col",
        sharey=True,
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.03, wspace=0.02, hspace=0.02)
    any_data = False
    plotted_L_values: set[int] = set()
    theory_plotted = False

    for row, bc in enumerate(("pbc", "obc")):
        colors = colors_for_bc(bc)
        for col, g_target in enumerate(g_targets):
            ax = axes[row, col]
            axins = ax.inset_axes([0.58, 0.56, 0.36, 0.34])
            panel_files = [
                f
                for f in files
                if f.bc == bc and f.g is not None and abs(f.g - g_target) < 5e-4
            ]
            data_curves: list[np.ndarray] = []
            inset_curves: list[tuple[np.ndarray, np.ndarray]] = []
            theta_max = 10.0
            kappa0 = 1.0
            kappa1 = -1.0

            for f in sorted(panel_files, key=lambda x: x.L):
                data = load_table(f.path, min_cols=16)
                if data is None:
                    continue
                theta = data[:, 0]
                y = data[:, 11]
                kappa0 = float(data[0, 8])
                kappa1 = float(data[0, 9])
                theta_max = max(theta_max, float(np.nanmax(theta)))
                data_curves.append(y)
                inset_curves.append((theta, y))
                plotted_L_values.add(f.L)
                ax.plot(
                    theta,
                    y,
                    "-",
                    color=colors[f.L],
                    lw=DATA_LINEWIDTH,
                    solid_capstyle="round",
                    solid_joinstyle="round",
                    antialiased=True,
                )
                axins.plot(
                    theta,
                    y,
                    "-",
                    color=colors[f.L],
                    lw=DATA_LINEWIDTH,
                    solid_capstyle="round",
                    solid_joinstyle="round",
                    antialiased=True,
                )
                any_data = True

            theta_grid = np.linspace(0.0, theta_max, 800)
            theory = align_two_level_sign(two_level_curve(theta_grid, kappa0, kappa1), data_curves)
            ax.plot(
                theta_grid,
                theory,
                "-",
                color="0.45",
                lw=REFERENCE_LINEWIDTH,
                alpha=0.75,
                solid_capstyle="round",
                solid_joinstyle="round",
                antialiased=True,
            )
            axins.plot(
                theta_grid,
                theory,
                "-",
                color="0.45",
                lw=REFERENCE_LINEWIDTH,
                alpha=0.75,
                solid_capstyle="round",
                solid_joinstyle="round",
                antialiased=True,
            )
            theory_plotted = True

            ax.axhline(0.0, color="0.55", lw=REFERENCE_LINEWIDTH, ls=":")
            ax.grid(alpha=0.25, ls=":")
            ax.set_xlim(0.0, theta_max)
            ax.set_ylim(-1.15, 1.15)
            ax.set_title(f"{label_for_bc(bc)} (g={g_target})", loc="right", fontsize=TITLE_FONTSIZE)
            ax.tick_params(labelsize=TICK_FONTSIZE, direction="in", which="both", top=True, right=True)
            if bc == "pbc" and abs(g_target - 0.5) < 5e-4:
                inset_xlim0, inset_xlim1, inset_ylim0, inset_ylim1 = (
                    inset_limits_near_theory_peak(
                        theta_grid,
                        theory,
                        inset_curves,
                        g_target,
                        half_width=0.15,
                        pad_frac=0.025,
                        min_yspan=0.02,
                    )
                )
            else:
                inset_xlim0, inset_xlim1, inset_ylim0, inset_ylim1 = (
                    inset_limits_near_theory_peak(
                        theta_grid,
                        theory,
                        inset_curves,
                        g_target,
                    )
                )
            axins.set_xlim(inset_xlim0, inset_xlim1)
            axins.set_ylim(inset_ylim0, inset_ylim1)
            axins.grid(alpha=0.25, ls=":")
            axins.tick_params(labelsize=10, direction="in", which="both", top=True, right=True)

    for ax in axes[0, :]:
        ax.tick_params(labelbottom=False)
        ax.set_xlabel("")
    for ax in axes[1, :]:
        ax.set_xlabel(r"$\Theta=\Delta_0 t$", fontsize=AXIS_LABEL_FONTSIZE)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$M_z(t)/m_0$", fontsize=AXIS_LABEL_FONTSIZE)
    for ax in axes[:, 1]:
        ax.tick_params(labelleft=False)
        ax.set_ylabel("")

    if any_data:
        extra_handles = None
        extra_labels = None
        if theory_plotted:
            extra_handles = [Line2D([], [], color="0.45", ls="-", lw=REFERENCE_LINEWIDTH)]
            extra_labels = ["two-level"]
        add_external_legend(
            fig,
            sorted(plotted_L_values),
            extra_handles=extra_handles,
            extra_labels=extra_labels,
        )
        out = PLOT_DIR / "foqt_dynamic_scaling.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"[OK] wrote {out}")
    plt.close(fig)


def format_phi(phi: float) -> str:
    if abs(phi - round(phi)) < 1e-9:
        return f"{int(round(phi))}"
    return f"{phi:g}"


def plot_loschmidt_echo(files: list[DataFile]) -> None:
    if not files:
        print("[WARN] no Loschmidt data found")
        return

    phi_targets = [1.0, 3.0]
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14.0, 10.0),
        sharex=True,
        sharey=False,
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.03, wspace=0.02, hspace=0.02)
    any_data = False
    plotted_L_values: set[int] = set()

    for row, bc in enumerate(("pbc", "obc")):
        colors = colors_for_bc(bc)
        for col, phi_target in enumerate(phi_targets):
            ax = axes[row, col]
            panel_files = [
                f
                for f in files
                if f.bc == bc and f.phi is not None and abs(f.phi - phi_target) < 5e-4
            ]

            for f in sorted(panel_files, key=lambda x: x.L):
                data = load_table(f.path, min_cols=12)
                if data is None:
                    continue
                theta = data[:, 0]
                q_echo = data[:, 9]
                ax.plot(
                    theta,
                    q_echo,
                    "-",
                    color=colors[f.L],
                    lw=DATA_LINEWIDTH,
                    solid_capstyle="round",
                    solid_joinstyle="round",
                    antialiased=True,
                )
                plotted_L_values.add(f.L)
                any_data = True

            ax.axhline(0.0, color="0.55", lw=REFERENCE_LINEWIDTH, ls=":")
            ax.grid(alpha=0.25, ls=":")
            ax.set_title(
                rf"{label_for_bc(bc)}, $\Phi={format_phi(phi_target)}$",
                loc="right",
                fontsize=TITLE_FONTSIZE,
            )
            ax.tick_params(labelsize=TICK_FONTSIZE, direction="in", which="both", top=True, right=True)

    for ax in axes[0, :]:
        ax.tick_params(labelbottom=False)
        ax.set_xlabel("")
    for ax in axes[1, :]:
        ax.set_xlabel(r"$\Theta=t/L^z$", fontsize=AXIS_LABEL_FONTSIZE)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$Q(t)=-\ln |\tilde A(t)|^2$", fontsize=AXIS_LABEL_FONTSIZE)
    for ax in axes[:, 1]:
        ax.set_ylabel("")

    if any_data:
        add_external_legend(fig, sorted(plotted_L_values))
        out = PLOT_DIR / "loschmidt_echo.pdf"
        fig.savefig(out, bbox_inches="tight")
        print(f"[OK] wrote {out}")
    else:
        print("[WARN] no usable Loschmidt tables found")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot quench ED scaling data.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    data_dir = args.data_dir
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    setup_matplotlib()

    cqt_files = discover_cqt(data_dir)
    foqt_files = discover_foqt(data_dir)
    loschmidt_files = discover_loschmidt(data_dir)

    if args.list:
        print(f"CQT files: {len(cqt_files)}")
        for f in cqt_files:
            print(f"  {f.bc} L={f.L}: {f.path}")
        print(f"FOQT files: {len(foqt_files)}")
        for f in foqt_files:
            print(f"  {f.bc} g={f.g:.3f} L={f.L}: {f.path}")
        print(f"Loschmidt files: {len(loschmidt_files)}")
        for f in loschmidt_files:
            print(f"  {f.bc} Phi={f.phi:.3f} L={f.L}: {f.path}")
        return

    plot_cqt(cqt_files)
    plot_foqt(foqt_files)
    plot_loschmidt_echo(loschmidt_files)


if __name__ == "__main__":
    main()
