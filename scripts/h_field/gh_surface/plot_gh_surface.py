#!/usr/bin/env python3
"""Plot two-parameter (g,h) surface outputs."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    mpl_cache_dir = Path("/tmp/qising_1d_matplotlib_cache/gh_surface")
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache_dir)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "h_field" / "gh_surface" / "raw"
PLOT_DIR = PROJECT_ROOT / "plots" / "hfield" / "gh_surface"
CONSTANTS_PATH = PROJECT_ROOT / "data" / "h_null" / "fss" / "fss_constants.json"

EXPECTED_COLUMNS = [
    "g",
    "h",
    "kappa_g",
    "kappa_h",
    "E0",
    "E1",
    "E2",
    "Delta0",
    "Delta1",
    "mz",
    "abs_mz",
    "mx",
    "method_code",
    "resid0",
    "resid1",
    "resid2",
]

plt.rcParams.update(
    {
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }
)


@dataclass
class Dataset:
    path: Path
    meta: dict[str, str]
    columns: dict[str, int]
    data: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot gh surface raw datasets.")
    parser.add_argument("--L", type=int, default=8)
    parser.add_argument("--pbc", type=int, choices=(0, 1), default=1)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--plot-dir", type=Path, default=PLOT_DIR)
    parser.add_argument("--constants", type=Path, default=CONSTANTS_PATH)
    parser.add_argument("--gap-log-color", action="store_true")
    parser.add_argument(
        "--grid-type",
        choices=("scaling", "physical", "auto"),
        default="scaling",
        help="Dataset grid to plot. Default: scaling.",
    )
    parser.add_argument(
        "--quantity-set",
        choices=("minimal", "all"),
        default="minimal",
        help="minimal writes only mz/gap for the selected grid. all may include optional physical plots.",
    )
    parser.add_argument(
        "--clean-minimal",
        action="store_true",
        help="Remove obsolete single-L PDFs for this L/BC, keeping only minimal outputs.",
    )
    return parser.parse_args()


def bc_name(pbc: int) -> str:
    return "PBC" if pbc else "OBC"


def bc_label(pbc: int) -> str:
    return "pbc" if pbc else "obc"


def load_constants(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        constants = json.load(fp)
    for bc in ("PBC", "OBC"):
        for key in ("g_pc", "beta_over_nu"):
            if key not in constants.get(bc, {}):
                raise KeyError(f"{path} {bc} entry is missing {key}")
    return constants


def parse_header(path: Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.startswith("#"):
                continue
            text = line[1:].strip()
            if "=" in text:
                key, value = text.split("=", 1)
                meta[key.strip()] = value.strip()
    return meta


def header_int(meta: dict[str, str], key: str) -> int | None:
    value = meta.get(key)
    if value is None:
        return None
    match = re.match(r"[-+]?\d+", value)
    if not match:
        return None
    return int(match.group(0))


def columns_from_meta(meta: dict[str, str]) -> dict[str, int]:
    raw = meta.get("columns")
    if raw is None:
        raise ValueError("raw header is missing columns")
    names = raw.split()
    if len(names) != len(EXPECTED_COLUMNS):
        raise ValueError(
            f"raw header has {len(names)} columns, expected {len(EXPECTED_COLUMNS)}"
        )
    missing = [name for name in EXPECTED_COLUMNS if name not in names]
    if missing:
        raise ValueError(f"raw header is missing columns: {', '.join(missing)}")
    return {name: names.index(name) for name in names}


def load_dataset(path: Path) -> Dataset:
    meta = parse_header(path)
    columns = columns_from_meta(meta)
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != len(EXPECTED_COLUMNS):
        raise ValueError(
            f"{path.relative_to(PROJECT_ROOT)} has {data.shape[1]} columns, "
            f"expected {len(EXPECTED_COLUMNS)}"
        )
    return Dataset(path=path, meta=meta, columns=columns, data=data)


def find_dataset(raw_dir: Path, grid_type: str, L: int, pbc: int) -> Dataset:
    exact = raw_dir / f"ghsurf_{grid_type}_{bc_label(pbc)}_L{L:02d}.dat"
    if exact.exists():
        return load_dataset(exact)

    matches: list[Path] = []
    for path in sorted(raw_dir.glob("*.dat")):
        meta = parse_header(path)
        if meta.get("grid_type") != grid_type:
            continue
        if header_int(meta, "L") != L:
            continue
        if header_int(meta, "pbc") != pbc:
            continue
        matches.append(path)
    if not matches:
        raise FileNotFoundError(
            f"no {grid_type} raw dataset found for L={L}, pbc={pbc} in "
            f"{raw_dir.relative_to(PROJECT_ROOT)}"
        )
    matches.sort(key=lambda p: p.stat().st_mtime_ns, reverse=True)
    return load_dataset(matches[0])


def maybe_find_dataset(raw_dir: Path, grid_type: str, L: int, pbc: int) -> Dataset | None:
    try:
        return find_dataset(raw_dir, grid_type, L, pbc)
    except FileNotFoundError:
        return None


def unique_sorted(values: np.ndarray) -> np.ndarray:
    return np.array(sorted(np.unique(values)), dtype=float)


def nearest_index(values: np.ndarray, value: float) -> int:
    idx = np.where(np.isclose(values, value, rtol=0.0, atol=1e-10))[0]
    if len(idx) != 1:
        raise ValueError(f"value {value:.17e} does not map uniquely to grid")
    return int(idx[0])


def rectangular_grid(ds: Dataset, x_name: str, y_name: str, z_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = ds.data[:, ds.columns[x_name]]
    y = ds.data[:, ds.columns[y_name]]
    xs = unique_sorted(x)
    ys = unique_sorted(y)
    z = np.full((len(ys), len(xs)), np.nan)
    filled = np.zeros_like(z, dtype=bool)
    for xv, yv, zv in zip(x, y, z_values):
        ix = nearest_index(xs, float(xv))
        iy = nearest_index(ys, float(yv))
        if filled[iy, ix]:
            raise ValueError(f"duplicate grid point x={xv:.17e}, y={yv:.17e}")
        z[iy, ix] = zv
        filled[iy, ix] = True
    if not np.all(filled):
        missing = int(np.size(filled) - np.count_nonzero(filled))
        raise ValueError(f"dataset is not a complete rectangular grid; missing={missing}")
    X, Y = np.meshgrid(xs, ys)
    return X, Y, z


def check_positive_gaps(name: str, ds: Dataset) -> list[str]:
    issues: list[str] = []
    for col in ("Delta0", "Delta1"):
        values = ds.data[:, ds.columns[col]]
        finite = values[np.isfinite(values)]
        if finite.size and np.any(finite <= 0.0):
            issues.append(f"{name}: {col} has non-positive finite values")
    return issues


def symmetry_checks(ds: Dataset) -> tuple[float, float, str]:
    g = ds.data[:, ds.columns["g"]]
    h = ds.data[:, ds.columns["h"]]
    mz = ds.data[:, ds.columns["mz"]]
    gap = ds.data[:, ds.columns["Delta0"]]
    values: dict[tuple[int, int], tuple[float, float]] = {}
    tol = 1e-10
    for gv, hv, mzv, gapv in zip(g, h, mz, gap):
        values[(int(round(gv / tol)), int(round(hv / tol)))] = (float(mzv), float(gapv))

    max_odd = 0.0
    max_even = 0.0
    compared = 0
    for gv, hv, mzv, gapv in zip(g, h, mz, gap):
        key = (int(round(gv / tol)), int(round(-hv / tol)))
        other = values.get(key)
        if other is None:
            continue
        max_odd = max(max_odd, abs(float(mzv) + other[0]))
        max_even = max(max_even, abs(float(gapv) - other[1]))
        compared += 1
    if compared == 0:
        return max_odd, max_even, "WARN"
    if max_odd < 1e-8 and max_even < 1e-8:
        return max_odd, max_even, "PASS"
    if max_odd < 1e-5 and max_even < 1e-5:
        return max_odd, max_even, "WARN"
    return max_odd, max_even, "FAIL"


def symmetry_checks_grid(
    ds: Dataset,
    x_name: str,
    y_name: str,
    mz_values: np.ndarray,
    gap_values: np.ndarray,
) -> tuple[float, float, str]:
    x = ds.data[:, ds.columns[x_name]]
    y = ds.data[:, ds.columns[y_name]]
    values: dict[tuple[int, int], tuple[float, float]] = {}
    tol = 1e-10
    for xv, yv, mzv, gapv in zip(x, y, mz_values, gap_values):
        values[(int(round(xv / tol)), int(round(yv / tol)))] = (float(mzv), float(gapv))

    max_odd = 0.0
    max_even = 0.0
    compared = 0
    for xv, yv, mzv, gapv in zip(x, y, mz_values, gap_values):
        other = values.get((int(round(xv / tol)), int(round(-yv / tol))))
        if other is None:
            continue
        max_odd = max(max_odd, abs(float(mzv) + other[0]))
        max_even = max(max_even, abs(float(gapv) - other[1]))
        compared += 1
    if compared == 0:
        return max_odd, max_even, "WARN"
    if max_odd < 1e-8 and max_even < 1e-8:
        return max_odd, max_even, "PASS"
    if max_odd < 1e-5 and max_even < 1e-5:
        return max_odd, max_even, "WARN"
    return max_odd, max_even, "FAIL"


def clean_minimal_outputs(plot_dir: Path, ltag: str, label: str, keep: set[Path]) -> list[Path]:
    removed: list[Path] = []
    for path in sorted(plot_dir.glob(f"*surface*_{ltag}_{label}.pdf")):
        if path in keep:
            continue
        path.unlink()
        removed.append(path)
    return removed


def set_3d_axes(ax: plt.Axes) -> None:
    ax.tick_params(labelsize=10, pad=2)
    ax.view_init(elev=26, azim=-135)


def add_centered_colorbar_top_label(
    fig: plt.Figure,
    colorbar: matplotlib.colorbar.Colorbar,
    label: str,
    *,
    pad: float = 0.012,
    fontsize: int = 14,
) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = colorbar.ax.get_tightbbox(renderer)
    bbox_fig = bbox.transformed(fig.transFigure.inverted())
    x_center = 0.5 * (bbox_fig.x0 + bbox_fig.x1)
    fig.text(
        x_center,
        bbox_fig.y1 + pad,
        label,
        ha="center",
        va="bottom",
        fontsize=fontsize,
    )


def plot_two_panel(
    *,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    x_label: str,
    y_label: str,
    z_label: str,
    title_left: str,
    title_right: str,
    cmap: str,
    out_path: Path,
    symmetric_color: bool = False,
    guide_x: float | None = None,
    guide_y: float | None = None,
    figure_title: str | None = None,
    figure_title_y: float = 0.985,
    show_surface_colorbar: bool = True,
    figsize: tuple[float, float] = (12.8, 5.4),
    width_ratios: tuple[float, float] = (1.02, 1.08),
    wspace: float = 0.22,
    contour_colorbar_pad: float = 0.04,
    contour_colorbar_label_side: str = "right",
    z_labelpad: float = 8.0,
    light_3d_background: bool = False,
    flip_zlabel: bool = False,
    zero_contour: bool = False,
) -> None:
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=width_ratios, wspace=wspace)
    if figure_title:
        fig.suptitle(figure_title, fontsize=17, y=figure_title_y)

    ax0 = fig.add_subplot(gs[0, 0])
    levels = 80
    norm = None
    if symmetric_color:
        vmax = float(np.nanmax(np.abs(Z)))
        if vmax > 0.0:
            norm = Normalize(vmin=-vmax, vmax=vmax)
    contour = ax0.contourf(X, Y, Z, levels=levels, cmap=cmap, norm=norm)
    ax0.contour(X, Y, Z, levels=12, colors="k", linewidths=0.35, alpha=0.35)
    if zero_contour and float(np.nanmin(Z)) <= 0.0 <= float(np.nanmax(Z)):
        ax0.contour(
            X,
            Y,
            Z,
            levels=[0.0],
            colors="0.08",
            linewidths=1.15,
            alpha=0.9,
        )
    if guide_x is not None:
        ax0.axvline(guide_x, color="white", linewidth=1.1, linestyle="--", alpha=0.95)
    if guide_y is not None:
        ax0.axhline(guide_y, color="white", linewidth=1.1, linestyle="--", alpha=0.95)
    ax0.set_xlabel(x_label)
    ax0.set_ylabel(y_label)
    if title_left:
        ax0.set_title(title_left)
    ax0.tick_params(direction="in", top=True, right=True)
    cb0 = fig.colorbar(contour, ax=ax0, fraction=0.046, pad=contour_colorbar_pad)
    top_label_colorbar = None
    if contour_colorbar_label_side == "top":
        top_label_colorbar = cb0
    else:
        cb0.set_label(z_label)
    if contour_colorbar_label_side == "left":
        cb0.ax.yaxis.set_label_position("left")
        cb0.ax.yaxis.set_ticks_position("right")

    ax1 = fig.add_subplot(gs[0, 1], projection="3d")
    surface = ax1.plot_surface(
        X,
        Y,
        Z,
        cmap=cmap,
        norm=norm,
        linewidth=0.0,
        antialiased=True,
        rcount=min(100, X.shape[0]),
        ccount=min(100, X.shape[1]),
    )
    ax1.set_xlabel(x_label, labelpad=8)
    ax1.set_ylabel(y_label, labelpad=8)
    ax1.set_zlabel(z_label, labelpad=z_labelpad)
    if title_right:
        ax1.set_title(title_right)
    set_3d_axes(ax1)
    if flip_zlabel:
        ax1.zaxis.set_rotate_label(False)
        ax1.zaxis.label.set_rotation(90)
    if light_3d_background:
        ax1.set_facecolor("white")
        for axis in (ax1.xaxis, ax1.yaxis, ax1.zaxis):
            axis.pane.set_facecolor((0.98, 0.98, 0.98, 1.0))
            axis.pane.set_edgecolor((0.86, 0.86, 0.86, 1.0))
            axis._axinfo["grid"]["color"] = (0.78, 0.78, 0.78, 0.75)
    if show_surface_colorbar:
        cb1 = fig.colorbar(surface, ax=ax1, fraction=0.046, pad=0.08)
        cb1.set_label(z_label)

    if top_label_colorbar is not None:
        add_centered_colorbar_top_label(fig, top_label_colorbar, z_label)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path.relative_to(PROJECT_ROOT)}")


def report_style_kwargs() -> dict:
    return {
        "figure_title_y": 0.955,
        "show_surface_colorbar": False,
        "figsize": (13.8, 6.2),
        "width_ratios": (1.06, 1.18),
        "wspace": 0.16,
        "contour_colorbar_pad": 0.025,
        "contour_colorbar_label_side": "top",
        "z_labelpad": 1.0,
        "light_3d_background": True,
        "flip_zlabel": True,
    }


def plot_surface_pair(
    *,
    ds: Dataset,
    grid_type: str,
    args: argparse.Namespace,
    bc: str,
    label: str,
    ltag: str,
    beta_over_nu: float,
    g_pc: float,
) -> tuple[list[Path], dict[str, object], list[str]]:
    issues: list[str] = []
    scale = float(args.L) ** beta_over_nu
    mz_values = ds.data[:, ds.columns["mz"]]
    gap_values = ds.data[:, ds.columns["Delta0"]]
    if np.any(gap_values[np.isfinite(gap_values)] <= 0.0):
        raise ValueError(f"{grid_type}: Delta0 has non-positive finite values")

    if grid_type == "scaling":
        x_name, y_name = "kappa_g", "kappa_h"
        x_label, y_label = r"$\kappa_g$", r"$\kappa_h$"
        guide_x, guide_y = 0.0, 0.0
        mz_plot_values = mz_values * scale
        mz_z_label = r"$m_z L^{\beta/\nu}$"
        mz_title = rf"$m_z L^{{\beta/\nu}}(\kappa_g,\kappa_h)$ - L={args.L} - {bc}"
        gap_title = rf"$\Delta_0(\kappa_g,\kappa_h)$ - L={args.L} - {bc}"
    else:
        x_name, y_name = "g", "h"
        x_label, y_label = r"$g$", r"$h$"
        guide_x, guide_y = g_pc, 0.0
        mz_plot_values = mz_values
        mz_z_label = r"$m_z$"
        mz_title = rf"$m_z(g,h)$ - L={args.L} - {bc}"
        gap_title = rf"$\Delta_0(g,h)$ - L={args.L} - {bc}"

    X, Y, Zmz = rectangular_grid(ds, x_name, y_name, mz_plot_values)
    _, _, Zgap_raw = rectangular_grid(ds, x_name, y_name, gap_values)
    Zgap_plot = Zgap_raw
    gap_z_label = r"$\Delta_0$"
    if args.gap_log_color:
        if np.any(gap_values <= 0.0):
            raise ValueError(f"{grid_type}: cannot use log10 color because Delta0 has non-positive values")
        Zgap_plot = np.log10(Zgap_raw)
        gap_z_label = r"$\log_{10}(\Delta_0)$"

    max_odd, max_even, symmetry_verdict = symmetry_checks_grid(
        ds, x_name, y_name, mz_plot_values, gap_values
    )

    out_mz = args.plot_dir / f"mz_surface_{grid_type}_{ltag}_{label}.pdf"
    out_gap = args.plot_dir / f"gap_surface_{grid_type}_{ltag}_{label}.pdf"
    style_kwargs = report_style_kwargs()
    plot_two_panel(
        X=X,
        Y=Y,
        Z=Zmz,
        x_label=x_label,
        y_label=y_label,
        z_label=mz_z_label,
        title_left="",
        title_right="",
        cmap="Blues",
        out_path=out_mz,
        symmetric_color=True,
        zero_contour=True,
        guide_x=guide_x,
        guide_y=guide_y,
        figure_title=mz_title,
        **style_kwargs,
    )
    plot_two_panel(
        X=X,
        Y=Y,
        Z=Zgap_plot,
        x_label=x_label,
        y_label=y_label,
        z_label=gap_z_label,
        title_left="",
        title_right="",
        cmap="Blues",
        out_path=out_gap,
        guide_x=guide_x,
        guide_y=guide_y,
        figure_title=gap_title,
        **style_kwargs,
    )

    if symmetry_verdict == "WARN":
        issues.append(f"{grid_type}: symmetry check WARN")
    elif symmetry_verdict == "FAIL":
        issues.append(f"{grid_type}: symmetry check FAIL")

    stats = {
        "grid_type": grid_type,
        "path": ds.path,
        "rows": int(ds.data.shape[0]),
        "columns": int(ds.data.shape[1]),
        "shape": Zgap_raw.shape,
        "x_range": (float(np.nanmin(X)), float(np.nanmax(X))),
        "y_range": (float(np.nanmin(Y)), float(np.nanmax(Y))),
        "delta0_min": float(np.nanmin(Zgap_raw)),
        "delta0_max": float(np.nanmax(Zgap_raw)),
        "mz_min": float(np.nanmin(Zmz)),
        "mz_max": float(np.nanmax(Zmz)),
        "max_odd": max_odd,
        "max_even": max_even,
        "symmetry_verdict": symmetry_verdict,
    }
    return [out_gap, out_mz], stats, issues


def main() -> None:
    args = parse_args()
    constants = load_constants(args.constants)
    bc = bc_name(args.pbc)
    label = bc_label(args.pbc)
    ltag = f"L{args.L:02d}"
    beta_over_nu = float(constants[bc]["beta_over_nu"])
    g_pc = float(constants[bc]["g_pc"])
    produced: list[Path] = []
    dataset_stats: list[dict[str, object]] = []
    issues: list[str] = []
    fail = False

    scaling = maybe_find_dataset(args.raw_dir, "scaling", args.L, args.pbc)
    physical = maybe_find_dataset(args.raw_dir, "physical", args.L, args.pbc)
    if physical is None and args.grid_type != "physical":
        issues.append(f"physical dataset missing for L={args.L}, {bc}; skipped")

    datasets_to_plot: list[tuple[str, Dataset]] = []
    if args.grid_type == "scaling":
        if scaling is None:
            print(f"  ERROR missing scaling dataset for L={args.L}, {bc}")
            raise SystemExit(1)
        datasets_to_plot.append(("scaling", scaling))
    elif args.grid_type == "physical":
        if physical is None:
            print(f"  ERROR missing physical dataset for L={args.L}, {bc}")
            raise SystemExit(1)
        datasets_to_plot.append(("physical", physical))
    else:
        if scaling is not None:
            datasets_to_plot.append(("scaling", scaling))
        elif physical is not None:
            datasets_to_plot.append(("physical", physical))
        else:
            print(f"  ERROR no scaling or physical dataset for L={args.L}, {bc}")
            raise SystemExit(1)

    if args.quantity_set == "all" and physical is not None:
        if not any(kind == "physical" for kind, _ in datasets_to_plot):
            datasets_to_plot.append(("physical", physical))

    for grid_type, dataset in datasets_to_plot:
        try:
            paths, stats, grid_issues = plot_surface_pair(
                ds=dataset,
                grid_type=grid_type,
                args=args,
                bc=bc,
                label=label,
                ltag=ltag,
                beta_over_nu=beta_over_nu,
                g_pc=g_pc,
            )
        except Exception as exc:
            print(f"  ERROR {grid_type}: {exc}")
            fail = True
            continue
        produced.extend(paths)
        dataset_stats.append(stats)
        issues.extend(grid_issues)

    minimal_keep = {
        args.plot_dir / f"gap_surface_scaling_{ltag}_{label}.pdf",
        args.plot_dir / f"mz_surface_scaling_{ltag}_{label}.pdf",
    }
    removed: list[Path] = []
    if args.clean_minimal:
        removed = clean_minimal_outputs(args.plot_dir, ltag, label, minimal_keep)
    else:
        obsolete = [
            path for path in sorted(args.plot_dir.glob(f"*surface*_{ltag}_{label}.pdf"))
            if path not in minimal_keep
        ]
        if obsolete and args.quantity_set == "minimal":
            issues.append(
                "--clean-minimal not set; obsolete single-L PDFs kept: "
                + ", ".join(path.name for path in obsolete)
            )

    missing_minimal = [path for path in sorted(minimal_keep) if not path.exists()]
    if args.quantity_set == "minimal" and args.grid_type == "scaling" and missing_minimal:
        fail = True
        issues.append("minimal scaling outputs missing: " + ", ".join(path.name for path in missing_minimal))

    print("[plot_gh_surface.py] final report")
    print(f"  L = {args.L}")
    print(f"  BC = {bc}")
    print(f"  grid_type = {args.grid_type}")
    print(f"  quantity_set = {args.quantity_set}")
    print(f"  beta_over_nu = {beta_over_nu:.12g}")
    print("  cmap gap = Blues")
    print("  cmap mz = Blues")
    print("  colorbar top label alignment = tightbbox/full extent")
    print("  normalization mz = symmetric around zero (Normalize, vmin=-max_abs, vmax=+max_abs)")
    print("  datasets:")
    for stats in dataset_stats:
        path = stats["path"]
        print(f"    {stats['grid_type']}: {path.relative_to(PROJECT_ROOT)}")
        print(f"      rows = {stats['rows']}")
        print(f"      columns = {stats['columns']}")
        print(f"      grid shape = {stats['shape']}")
        print(f"      range x = [{stats['x_range'][0]:.6g}, {stats['x_range'][1]:.6g}]")
        print(f"      range y = [{stats['y_range'][0]:.6g}, {stats['y_range'][1]:.6g}]")
        print(f"      min/max Delta0 = [{stats['delta0_min']:.6e}, {stats['delta0_max']:.6e}]")
        print(f"      min/max mz scaled = [{stats['mz_min']:.6e}, {stats['mz_max']:.6e}]")
        print(f"      max oddness mz = {stats['max_odd']:.6e}")
        print(f"      max even mismatch Delta0 = {stats['max_even']:.6e}")
        print(f"      symmetry verdict = {stats['symmetry_verdict']}")
    print("  PDF prodotti:")
    for path in produced:
        print(f"    {path.relative_to(PROJECT_ROOT)}")
    print(f"  PDF prodotti dal run minimal = {len(produced) if args.quantity_set == 'minimal' else 'n/a'}")
    if removed:
        print("  PDF rimossi da --clean-minimal:")
        for path in removed:
            print(f"    {path.relative_to(PROJECT_ROOT)}")
    for issue in issues:
        print(f"  WARN {issue}")

    if fail or any("FAIL" in issue for issue in issues):
        verdict = "FAIL"
    elif issues:
        verdict = "WARN"
    else:
        verdict = "PASS"
    print(f"  plot_verdict = {verdict}")
    if verdict == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
