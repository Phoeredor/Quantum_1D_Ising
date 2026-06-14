#!/usr/bin/env python3
"""
Report-ready plots for longitudinal h-field data.

Inputs are read from data/h_field/hfield_raw. Figures are written directly
under plots/hfield. Derived longitudinal susceptibility tables are written
to data/h_field/hfield_processed.

Usage: python3 scripts/h_field/plot_hfield.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    mpl_cache_dir = Path("/tmp/qising_1d_matplotlib_cache/h_field")
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache_dir)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import CubicSpline, PchipInterpolator


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "h_field" / "hfield_raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "h_field" / "hfield_processed"
FSS_CONSTANTS_PATH = PROJECT_ROOT / "data" / "h_null" / "fss" / "fss_constants.json"
PLOT_DIR = PROJECT_ROOT / "plots" / "hfield"
LEGACY_PLOT_DIRS = [
    PLOT_DIR / "report",
    PLOT_DIR / "CQT",
    PLOT_DIR / "FOQT",
    PLOT_DIR / "FSS",
]
for _path in (PROCESSED_DIR, PLOT_DIR):
    _path.mkdir(parents=True, exist_ok=True)

NUMERIC_OUTPUT_SUFFIXES = {".dat", ".txt", ".csv", ".json", ".npy", ".npz"}
FINAL_PDF_NAMES = [
    "cqt_order_parameter.pdf",
    "cqt_susceptibility.pdf",
    "foqt_scaling.pdf",
    "gaps_vs_h.pdf",
]
FINAL_PDF_PATHS = {PLOT_DIR / name for name in FINAL_PDF_NAMES}

FONT_SCALE = 1.6
AXIS_LABEL_FONTSIZE = 18
TICK_FONTSIZE = 15
LEGEND_FONTSIZE = 15
TITLE_FONTSIZE = 15
TEXT_FONTSIZE = 15
INSET_TICK_FONTSIZE = 12
LINEWIDTH_OFFSET = 0.0

# Report-sized figure to match gap_vs_h_ext readability.
FOQT_REPORT_FIGURE_WIDTH_IN = 10.0
FOQT_REPORT_FIGURE_HEIGHT_IN = 13.6
FOQT_PANEL_HSPACE = 0.30
FOQT_PANEL_WSPACE = 0.78
FOQT_LEGEND_HSPACE = 0.09
FOQT_LEGEND_HEIGHT_RATIO = 0.52

ED_SIZES = [4, 6, 8, 10, 12]
LZ_SIZES = [14, 16, 18, 20, 22]
ALL_SIZES = ED_SIZES + LZ_SIZES
EXCLUDED_SIZES_BY_BC = {
    "OBC": {22},
}

_cmap_pbc = plt.cm.plasma
_cmap_obc = plt.cm.viridis
_color_values = np.linspace(0.0, 0.85, len(ALL_SIZES))
COLORS_PBC = {L: _cmap_pbc(v) for L, v in zip(ALL_SIZES, _color_values)}
COLORS_OBC = {L: _cmap_obc(v) for L, v in zip(ALL_SIZES, _color_values)}
PBC_ACCENT = _cmap_pbc(0.65)
OBC_ACCENT = _cmap_obc(0.65)

HCOL = dict(
    h=0,
    scale_x=1,
    kappa=2,
    E0=3,
    E1=4,
    E2=5,
    E3=6,
    delta_h=7,
    delta0_h0=8,
    mz=9,
    abs_mz=10,
    mx=11,
    method_code=12,
    resid0=13,
    resid1=14,
    resid2=15,
    resid3=16,
)

DEFAULT_SIZES = ALL_SIZES
ROW_NCOL = 17
CQT_G_TOL = 1e-8
FOQT_G_TOL = 1e-10

WARNINGS: list[str] = []
CRITICAL_ERRORS: list[str] = []
SAVED_PDFS: list[Path] = []
SAVED_TABLES: list[Path] = []
FIGURE_RESULTS: list[tuple[str, bool, str]] = []
OBSOLETE_REMOVED: list[Path] = []
REMOVED_DIRS: list[Path] = []


@dataclass
class HFieldDataset:
    path: Path
    meta: dict[str, str]
    L: int
    g: float
    bc_label: str
    pbc: bool
    mode: str
    yh: float
    xmax: float
    dx_near: float
    dx_mid: float
    dx_far: float
    N_h: int | None
    delta0_h0: float
    m0: float
    method_code: int
    data: np.ndarray
    complete: bool
    partial: bool
    n_raw_rows: int
    n_valid_rows: int


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def warn(message: str) -> None:
    WARNINGS.append(message)
    print(f"  WARN {message}")


def critical(message: str) -> None:
    CRITICAL_ERRORS.append(message)
    print(f"  ERROR {message}")


def record_figure(name: str, produced: bool, reason: str = "") -> None:
    FIGURE_RESULTS.append((name, produced, reason))


def clean_hfield_pdf_outputs() -> list[Path]:
    removed: list[Path] = []
    if not PLOT_DIR.exists():
        return removed
    for path in sorted(PLOT_DIR.rglob("*.pdf")):
        if path in FINAL_PDF_PATHS:
            continue
        try:
            path.unlink()
            removed.append(path)
        except OSError as exc:
            warn(f"obsolete cleanup failed for {path}: {exc}")

    if removed:
        for path in removed:
            print(f"  [RM] {path.relative_to(PROJECT_ROOT)}")
    else:
        print("  [RM] no obsolete PDFs removed")

    for legacy_dir in LEGACY_PLOT_DIRS:
        try:
            legacy_dir.rmdir()
        except OSError:
            continue
        else:
            REMOVED_DIRS.append(legacy_dir)
            print(f"  [RM] empty dir {legacy_dir.relative_to(PROJECT_ROOT)}")

    OBSOLETE_REMOVED.extend(removed)
    return removed


def colors_for_bc(bc_label: str) -> dict[int, tuple]:
    return COLORS_PBC if bc_label.upper() == "PBC" else COLORS_OBC


def include_size_for_bc(bc_label: str, L: int) -> bool:
    return int(L) not in EXCLUDED_SIZES_BY_BC.get(bc_label.upper(), set())


def include_hfield_path(path: Path) -> bool:
    name = path.name.lower()
    return not (name.startswith("hfield_obc_") and name.endswith("_l22.dat"))


def apply_grid(ax: plt.Axes) -> None:
    ax.grid(True, which="major", axis="both",
            color="0.70", alpha=0.55,
            linestyle=":", linewidth=0.8 + LINEWIDTH_OFFSET, zorder=0)


def apply_ticks(ax: plt.Axes) -> None:
    ax.tick_params(direction="in", which="both",
                   top=True, right=True,
                   labelsize=TICK_FONTSIZE)


def autoscale_independent(ax: plt.Axes, *, xmargin=0.04, ymargin=0.08) -> None:
    ax.relim()
    ax.autoscale_view()
    ax.margins(x=xmargin, y=ymargin)


def save_fig(fig: plt.Figure, name: str) -> None:
    path = PLOT_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    SAVED_PDFS.append(path)
    print(f"  [OK] {path.relative_to(PROJECT_ROOT)}")


def snapshot_outputs() -> dict[Path, tuple[int, int]]:
    snapshot: dict[Path, tuple[int, int]] = {}
    for root in (PLOT_DIR, PROCESSED_DIR):
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file():
                st = path.stat()
                snapshot[path] = (st.st_size, st.st_mtime_ns)
    return snapshot


def report_output_changes(before: dict[Path, tuple[int, int]]) -> None:
    changed: list[tuple[Path, str, int | None, int]] = []
    for root in (PLOT_DIR, PROCESSED_DIR):
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            st = path.stat()
            after = (st.st_size, st.st_mtime_ns)
            old = before.get(path)
            if old is None:
                changed.append((path, "created", None, st.st_size))
            elif old != after:
                changed.append((path, "updated", old[0], st.st_size))

    pdfs = [item for item in changed if item[0].suffix.lower() == ".pdf"]
    numeric = [item for item in changed if item[0].suffix.lower() in NUMERIC_OUTPUT_SUFFIXES]

    print("\n[plot_hfield.py]  Output before/after summary:")
    if pdfs:
        print("  PDF outputs changed:")
        for path, status, old_size, new_size in pdfs:
            rel = path.relative_to(PROJECT_ROOT)
            if old_size is None:
                print(f"    {status:7s} {rel} size={new_size} B")
            else:
                print(f"    {status:7s} {rel} size={old_size} -> {new_size} B")
    else:
        print("  PDF outputs changed: none")

    if numeric:
        print("  Numeric outputs changed:")
        for path, status, old_size, new_size in numeric:
            rel = path.relative_to(PROJECT_ROOT)
            if old_size is None:
                print(f"    {status:7s} {rel} size={new_size} B")
            else:
                print(f"    {status:7s} {rel} size={old_size} -> {new_size} B")
    else:
        print("  Numeric outputs changed: none")


def _as_float(value: str | None, default: float = np.nan) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: str | None, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _finite_range(values: np.ndarray) -> str:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return "[nan, nan]"
    return f"[{np.min(arr):.6e}, {np.max(arr):.6e}]"


def finite_minmax(values) -> tuple[float, float]:
    if isinstance(values, np.ndarray):
        arr = np.asarray(values, dtype=float).ravel()
    else:
        pieces = []
        for item in values:
            arr_item = np.asarray(item, dtype=float).ravel()
            if arr_item.size:
                pieces.append(arr_item)
        arr = np.concatenate(pieces) if pieces else np.asarray([], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    return float(np.min(arr)), float(np.max(arr))


def _method_label(method_code: int) -> str:
    if method_code == 0:
        return "ED"
    if method_code == 1:
        return "Lanczos"
    return f"method_code={method_code}"


def _line_style(dataset: HFieldDataset) -> str:
    if int(dataset.method_code) == 0:
        return "-"
    if int(dataset.method_code) == 1:
        return "--"
    return "-"


def _line_alpha(dataset: HFieldDataset) -> float:
    return 0.62 if dataset.partial else 1.0


def _line_label(dataset: HFieldDataset) -> str:
    label = rf"$L={dataset.L}$"
    if dataset.partial:
        label += " partial"
    return label


def _legend_for_axis(ax: plt.Axes, datasets: list[HFieldDataset],
                     *, loc: str = "best", frameon: bool = False,
                     ncol: int | None = None) -> None:
    from matplotlib.lines import Line2D

    handles, labels = ax.get_legend_handles_labels()
    seen: set[str] = set()
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels):
        if not label or label == "_nolegend_" or label in seen:
            continue
        seen.add(label)
        unique_handles.append(handle)
        unique_labels.append(label)

    if any(ds.method_code == 0 for ds in datasets):
        unique_handles.append(Line2D([], [], color="gray", ls="-", lw=1.7))
        unique_labels.append("ED")
    if any(ds.method_code == 1 for ds in datasets):
        unique_handles.append(Line2D([], [], color="gray", ls="--", lw=1.7))
        unique_labels.append("LNCZ")

    if not unique_handles:
        return
    if ncol is None:
        ncol = 2 if len(unique_handles) > 6 else 1
    legend = ax.legend(
        unique_handles, unique_labels,
        frameon=frameon,
        fontsize=LEGEND_FONTSIZE,
        ncol=ncol,
        loc=loc,
    )
    if frameon:
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_alpha(0.78)
        legend.get_frame().set_edgecolor("none")


def _size_legend(
    ax: plt.Axes,
    *,
    loc: str,
    ncol: int,
    bbox_to_anchor=None,
) -> None:
    handles, labels = ax.get_legend_handles_labels()
    seen: set[str] = set()
    size_handles = []
    size_labels = []
    for handle, label in zip(handles, labels):
        if not label.startswith("$L=") or label in seen:
            continue
        seen.add(label)
        size_handles.append(handle)
        size_labels.append(label)
    if not size_handles:
        return
    legend = ax.legend(
        size_handles,
        size_labels,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        ncol=ncol,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
    )
    ax.add_artist(legend)


def _method_theory_legend(
    ax: plt.Axes,
    datasets: list[HFieldDataset],
    *,
    loc: str = "lower right",
    ncol: int = 1,
    theory_handles=None,
) -> None:
    from matplotlib.lines import Line2D

    handles = []
    labels = []
    if any(ds.method_code == 0 for ds in datasets):
        handles.append(Line2D([], [], color="gray", ls="-", lw=1.7))
        labels.append("ED")
    if any(ds.method_code == 1 for ds in datasets):
        handles.append(Line2D([], [], color="gray", ls="--", lw=1.7))
        labels.append("LNCZ")
    for handle in theory_handles or []:
        label = handle.get_label()
        if label and label != "_nolegend_":
            handles.append(handle)
            labels.append(label)
    if not handles:
        return
    legend = ax.legend(
        handles,
        labels,
        frameon=True,
        fontsize=LEGEND_FONTSIZE,
        ncol=ncol,
        loc=loc,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.88)
    legend.get_frame().set_edgecolor("none")


def _theory_relation_legend(
    ax: plt.Axes,
    theory_handles,
    *,
    loc: str = "lower right",
    ncol: int = 1,
    fontsize: float | None = None,
) -> None:
    handles = []
    labels = []
    for handle in theory_handles or []:
        label = handle.get_label()
        if label and label != "_nolegend_":
            handles.append(handle)
            labels.append(label)
    if not handles:
        return
    legend = ax.legend(
        handles,
        labels,
        frameon=True,
        fontsize=fontsize if fontsize is not None else LEGEND_FONTSIZE,
        ncol=ncol,
        loc=loc,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.88)
    legend.get_frame().set_edgecolor("none")


class _DoubleBCLine:
    def __init__(self, L: int, linestyle: str = "-") -> None:
        self.L = int(L)
        self.linestyle = linestyle


class _DoubleBCLineHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        from matplotlib.lines import Line2D

        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        y_pbc = y0 + 0.68 * height
        y_obc = y0 + 0.32 * height
        artists = [
            Line2D(
                [x0, x0 + width], [y_pbc, y_pbc],
                color=COLORS_PBC.get(orig_handle.L, "gray"),
                ls=orig_handle.linestyle,
                lw=1.9 + LINEWIDTH_OFFSET,
                solid_capstyle="round",
            ),
            Line2D(
                [x0, x0 + width], [y_obc, y_obc],
                color=COLORS_OBC.get(orig_handle.L, "gray"),
                ls=orig_handle.linestyle,
                lw=1.9 + LINEWIDTH_OFFSET,
                solid_capstyle="round",
            ),
        ]
        for artist in artists:
            artist.set_transform(handlebox.get_transform())
            handlebox.add_artist(artist)
        return artists[0]


def _register_external_legend_dataset(
    ds: HFieldDataset,
    plotted_L_methods: dict[int, set[int]],
    plotted_methods: set[int],
) -> None:
    method_code = int(ds.method_code)
    plotted_L_methods.setdefault(int(ds.L), set()).add(method_code)
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
    y_anchor: float = -0.07,
    fontsize: float | None = None,
) -> None:
    from matplotlib.lines import Line2D

    handles = []
    labels = []
    for L in ALL_SIZES:
        if L not in plotted_L_methods:
            continue
        linestyle = _linestyle_for_method_set(plotted_L_methods[L]) if encode_method_in_size else "-"
        handles.append(_DoubleBCLine(L, linestyle))
        labels.append(rf"$L={L}$")

    if 0 in plotted_methods:
        handles.append(Line2D([], [], color="gray", ls="-", lw=1.9 + LINEWIDTH_OFFSET))
        labels.append("ED")
    if 1 in plotted_methods:
        handles.append(Line2D([], [], color="gray", ls="--", lw=1.9 + LINEWIDTH_OFFSET))
        labels.append("LNCZ")

    if not handles:
        return

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, y_anchor),
        bbox_transform=fig.transFigure,
        frameon=False,
        fontsize=fontsize if fontsize is not None else LEGEND_FONTSIZE,
        ncol=6,
        columnspacing=1.7,
        handlelength=2.2,
        handletextpad=0.8,
        handler_map={_DoubleBCLine: _DoubleBCLineHandler()},
    )


def _no_data(ax: plt.Axes, text: str = "No finite data") -> None:
    ax.text(0.5, 0.5, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=TEXT_FONTSIZE,
            color="gray")


def _target_g_label(g_value: float) -> str:
    return np.format_float_positional(float(g_value), trim="-", precision=10)


def _target_g_filename(g_value: float) -> str:
    return _target_g_label(g_value).replace("-", "m")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def read_header(path: Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    try:
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                stripped = line.strip()
                if not stripped:
                    continue
                if not stripped.startswith("#"):
                    continue
                body = stripped[1:].strip()
                if "=" not in body:
                    continue
                key, value = body.split("=", 1)
                meta[key.strip()] = value.strip()
    except OSError as exc:
        warn(f"{path.name}: cannot read header: {exc}")
    return meta


def _load_numeric_rows(path: Path) -> tuple[np.ndarray, int]:
    rows: list[list[float]] = []
    n_raw = 0
    try:
        with path.open("r", encoding="utf-8") as fp:
            for lineno, line in enumerate(fp, start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                n_raw += 1
                parts = stripped.split()
                if len(parts) != ROW_NCOL:
                    warn(
                        f"{path.name}:{lineno}: expected {ROW_NCOL} columns, "
                        f"got {len(parts)}; row skipped"
                    )
                    continue
                try:
                    rows.append([float(part) for part in parts])
                except ValueError:
                    warn(f"{path.name}:{lineno}: non-numeric row skipped")
    except OSError as exc:
        warn(f"{path.name}: cannot read data rows: {exc}")
    if not rows:
        return np.empty((0, ROW_NCOL), dtype=float), n_raw
    return np.asarray(rows, dtype=float), n_raw


def _prevalent_method_code(data: np.ndarray, path: Path) -> int:
    if data.size == 0:
        return -1
    col = data[:, HCOL["method_code"]]
    col = col[np.isfinite(col)]
    if col.size == 0:
        return -1
    vals, counts = np.unique(col.astype(int), return_counts=True)
    if vals.size > 1:
        warn(f"{path.name}: multiple method_code values {vals.tolist()}; using prevalent")
    return int(vals[np.argmax(counts)])


def _deduplicate_primary(data: np.ndarray, mode: str, path: Path) -> np.ndarray:
    if data.shape[0] <= 1:
        return data
    primary_col = HCOL["scale_x"] if mode == "cqt" else HCOL["kappa"]
    primary = data[:, primary_col]
    finite = np.isfinite(primary)
    if np.count_nonzero(finite) <= 1:
        return data[np.argsort(data[:, HCOL["h"]])]

    order = np.lexsort((data[:, HCOL["h"]], primary))
    sorted_data = data[order]
    sorted_primary = sorted_data[:, primary_col]
    keep = np.ones(sorted_data.shape[0], dtype=bool)
    last = np.nan
    duplicates = 0
    for i, value in enumerate(sorted_primary):
        if not np.isfinite(value):
            continue
        if np.isfinite(last) and abs(value - last) <= 1e-10:
            keep[i] = False
            duplicates += 1
            continue
        last = value
    if duplicates:
        warn(
            f"{path.name}: {duplicates} duplicate {('scale_x' if mode == 'cqt' else 'kappa')} "
            "rows within 1e-10; kept first after sort"
        )
    deduped = sorted_data[keep]
    return deduped[np.argsort(deduped[:, HCOL["h"]])]


def load_hfield_file(path: Path, *, allow_partial: bool = False) -> HFieldDataset | None:
    meta = read_header(path)
    data, n_raw = _load_numeric_rows(path)
    if data.shape[1] != ROW_NCOL:
        warn(f"{path.name}: no valid {ROW_NCOL}-column data rows")
        return None

    L = _as_int(meta.get("L"))
    g = _as_float(meta.get("g"))
    pbc_int = _as_int(meta.get("pbc"))
    mode = str(meta.get("mode", "")).strip().lower()
    N_h = _as_int(meta.get("N_h"))

    missing = []
    if L is None:
        missing.append("L")
    if not np.isfinite(g):
        missing.append("g")
    if pbc_int not in (0, 1):
        missing.append("pbc")
    if mode not in ("cqt", "foqt"):
        missing.append("mode")
    if missing:
        warn(f"{path.name}: missing/invalid header keys {missing}; skipped")
        return None

    data = data[np.isfinite(data[:, HCOL["h"]])]
    mode = str(mode)
    data = _deduplicate_primary(data, mode, path)
    n_valid = int(data.shape[0])

    if N_h is None:
        warn(f"{path.name}: header has no N_h")
        complete = False
        if not allow_partial:
            warn(f"{path.name}: skipped because N_h is missing (use --allow-partial)")
            return None
    else:
        complete = bool(n_valid == int(N_h))
        if not complete and not allow_partial:
            warn(
                f"{path.name}: partial {n_valid}/{N_h}; skipped "
                "(use --allow-partial)"
            )
            return None

    method_code = _prevalent_method_code(data, path)
    ds = HFieldDataset(
        path=path,
        meta=meta,
        L=int(L),
        g=float(g),
        bc_label="PBC" if int(pbc_int) == 1 else "OBC",
        pbc=bool(int(pbc_int)),
        mode=mode,
        yh=_as_float(meta.get("yh")),
        xmax=_as_float(meta.get("xmax")),
        dx_near=_as_float(meta.get("dx_near")),
        dx_mid=_as_float(meta.get("dx_mid")),
        dx_far=_as_float(meta.get("dx_far")),
        N_h=N_h,
        delta0_h0=_as_float(meta.get("delta0_h0")),
        m0=_as_float(meta.get("m0")),
        method_code=method_code,
        data=data,
        complete=complete,
        partial=not complete,
        n_raw_rows=int(n_raw),
        n_valid_rows=n_valid,
    )
    if ds.partial:
        expected = "unknown" if ds.N_h is None else str(ds.N_h)
        warn(f"{ds.path.name}: loaded partial dataset rows={ds.n_valid_rows}/{expected}")
    print_dataset_summary(ds)
    print_symmetry_diagnostics(ds)
    return ds


def discover_hfield_files(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        warn(f"raw directory does not exist: {raw_dir}")
        return []
    return sorted(raw_dir.glob("hfield_*.dat"))


def load_fss_constants(path: Path = FSS_CONSTANTS_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing FSS constants: {path}")
    with path.open("r", encoding="utf-8") as fp:
        constants = json.load(fp)
    for bc in ("PBC", "OBC"):
        if bc not in constants:
            raise KeyError(f"missing {bc} in {path}")
        for key in ("g_pc", "beta_over_nu", "gamma_over_nu", "y_h", "delta"):
            if key not in constants[bc]:
                raise KeyError(f"missing {bc}.{key} in {path}")
    return constants


def _rows_distance_from_expected(ds: HFieldDataset) -> int:
    if ds.N_h is None:
        return 10**9
    return abs(int(ds.n_valid_rows) - int(ds.N_h))


def _select_best_cqt(items: list[HFieldDataset], constants: dict) -> HFieldDataset:
    def key(ds: HFieldDataset):
        return (_rows_distance_from_expected(ds), -ds.n_valid_rows, -ds.path.stat().st_mtime_ns)

    chosen = sorted(items, key=key)[0]
    discarded = [ds for ds in items if ds is not chosen]
    if discarded:
        warn(
            f"CQT duplicate {chosen.bc_label} L={chosen.L}: kept {chosen.path.name}; "
            "discarded " + ", ".join(ds.path.name for ds in discarded)
        )
    return chosen


def _select_best_foqt(items: list[HFieldDataset], target_g: float) -> HFieldDataset:
    def key(ds: HFieldDataset):
        return (_rows_distance_from_expected(ds), -ds.n_valid_rows, -ds.path.stat().st_mtime_ns)

    chosen = sorted(items, key=key)[0]
    discarded = [ds for ds in items if ds is not chosen]
    if discarded:
        warn(
            f"FOQT duplicate {chosen.bc_label} g={_target_g_label(target_g)} L={chosen.L}: "
            f"kept {chosen.path.name}; discarded " + ", ".join(ds.path.name for ds in discarded)
        )
    return chosen


def load_all_hfield_data(
    *,
    raw_dir: Path,
    constants: dict,
    sizes: list[int],
    foqt_g: list[float],
    allow_partial: bool,
    do_cqt: bool,
    do_foqt: bool,
) -> dict:
    files = discover_hfield_files(raw_dir)
    print(f"[plot_hfield.py]  Discovered {len(files)} h-field data files in {raw_dir.relative_to(PROJECT_ROOT)}")

    loaded: list[HFieldDataset] = []
    for path in files:
        if not include_hfield_path(path):
            continue
        ds = load_hfield_file(path, allow_partial=allow_partial)
        if ds is None:
            continue
        loaded.append(ds)

    size_set = {int(L) for L in sizes}
    cqt_candidates: dict[tuple[str, int], list[HFieldDataset]] = {}
    foqt_candidates: dict[tuple[str, float, int], list[HFieldDataset]] = {}

    for ds in loaded:
        if ds.L not in size_set:
            continue
        if not include_size_for_bc(ds.bc_label, ds.L):
            continue
        if ds.mode == "cqt":
            if not do_cqt:
                continue
            g_pc = float(constants[ds.bc_label]["g_pc"])
            if abs(ds.g - g_pc) > CQT_G_TOL:
                warn(
                    f"{ds.path.name}: CQT g={ds.g:.12g} incompatible with "
                    f"{ds.bc_label} g_pc={g_pc:.12g}; skipped"
                )
                continue
            cqt_candidates.setdefault((ds.bc_label, ds.L), []).append(ds)
        elif ds.mode == "foqt":
            if not do_foqt:
                continue
            matches = [float(g) for g in foqt_g if abs(ds.g - float(g)) <= FOQT_G_TOL]
            if not matches:
                warn(
                    f"{ds.path.name}: FOQT g={ds.g:.12g} not in requested "
                    f"{[_target_g_label(g) for g in foqt_g]}; skipped"
                )
                continue
            target_g = sorted(matches, key=lambda val: abs(ds.g - val))[0]
            foqt_candidates.setdefault((ds.bc_label, target_g, ds.L), []).append(ds)

    cqt: dict[str, list[HFieldDataset]] = {"PBC": [], "OBC": []}
    for key, items in sorted(cqt_candidates.items()):
        chosen = _select_best_cqt(items, constants)
        cqt[key[0]].append(chosen)
    for bc in cqt:
        cqt[bc].sort(key=lambda ds: ds.L)

    foqt: dict[float, dict[str, list[HFieldDataset]]] = {
        float(g): {"PBC": [], "OBC": []} for g in foqt_g
    }
    for key, items in sorted(foqt_candidates.items()):
        bc, target_g, _L = key
        chosen = _select_best_foqt(items, target_g)
        foqt[target_g][bc].append(chosen)
    for target_g in foqt:
        for bc in ("PBC", "OBC"):
            foqt[target_g][bc].sort(key=lambda ds: ds.L)

    _report_missing_selection(cqt, foqt, sizes, foqt_g, do_cqt, do_foqt)
    return dict(all_loaded=loaded, cqt=cqt, foqt=foqt, discovered=files)


def _report_missing_selection(cqt: dict, foqt: dict, sizes: list[int],
                              foqt_g: list[float], do_cqt: bool, do_foqt: bool) -> None:
    if do_cqt:
        for bc in ("PBC", "OBC"):
            present = {ds.L for ds in cqt.get(bc, [])}
            missing = [L for L in sizes if include_size_for_bc(bc, L) and L not in present]
            if missing:
                warn(f"CQT {bc}: no selected complete dataset for L={missing}")
    if do_foqt:
        for target_g in foqt_g:
            for bc in ("PBC", "OBC"):
                present = {ds.L for ds in foqt.get(float(target_g), {}).get(bc, [])}
                missing = [L for L in sizes if include_size_for_bc(bc, L) and L not in present]
                if missing:
                    warn(f"FOQT {bc} g={_target_g_label(target_g)}: no selected complete dataset for L={missing}")


def print_dataset_summary(ds: HFieldDataset) -> None:
    primary_col = HCOL["scale_x"] if ds.mode == "cqt" else HCOL["kappa"]
    primary_name = "scale_x" if ds.mode == "cqt" else "kappa"
    status = "complete" if ds.complete else "partial"
    expected = "unknown" if ds.N_h is None else str(ds.N_h)
    print(
        f"  LOAD {ds.path.relative_to(PROJECT_ROOT)} | "
        f"L={ds.L:2d} {ds.bc_label} mode={ds.mode} g={ds.g:.12g} "
        f"N_h={expected} rows={ds.n_valid_rows} method={_method_label(ds.method_code)} "
        f"{status} h={_finite_range(ds.data[:, HCOL['h']])} "
        f"{primary_name}={_finite_range(ds.data[:, primary_col])}"
    )


def _paired_values(x: np.ndarray, y: np.ndarray, *, tol: float) -> list[tuple[float, float, float]]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size == 0:
        return []
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    pairs = []
    for xi, yi in zip(x, y):
        j = int(np.argmin(np.abs(x + xi)))
        if abs(x[j] + xi) <= tol:
            pairs.append((float(xi), float(yi), float(y[j])))
    return pairs


def _max_oddness(x: np.ndarray, y: np.ndarray, *, tol: float = 1e-12) -> float:
    pairs = _paired_values(x, y, tol=tol)
    if not pairs:
        return np.nan
    return float(max(abs(yp + ym) for _x, yp, ym in pairs))


def _max_even_mismatch(x: np.ndarray, y: np.ndarray, *, tol: float = 1e-12) -> float:
    pairs = _paired_values(x, y, tol=tol)
    if not pairs:
        return np.nan
    return float(max(abs(yp - ym) for _x, yp, ym in pairs))


def print_symmetry_diagnostics(ds: HFieldDataset) -> None:
    variable = ds.data[:, HCOL["h"]] if ds.mode == "cqt" else ds.data[:, HCOL["kappa"]]
    variable_name = "h" if ds.mode == "cqt" else "kappa"
    tol = 1e-10 if ds.method_code == 0 else 1e-8
    mz_odd = _max_oddness(variable, ds.data[:, HCOL["mz"]])
    gap_even = _max_even_mismatch(variable, ds.data[:, HCOL["delta_h"]])
    mx_even = _max_even_mismatch(variable, ds.data[:, HCOL["mx"]])
    print(
        f"       symmetry({variable_name}): "
        f"max|mz(x)+mz(-x)|={mz_odd:.3e}, "
        f"max|Delta(x)-Delta(-x)|={gap_even:.3e}, "
        f"max|mx(x)-mx(-x)|={mx_even:.3e}"
    )
    if np.isfinite(mz_odd) and mz_odd > tol:
        warn(f"{ds.path.name}: mz oddness {mz_odd:.3e} exceeds tolerance {tol:.1e}")
    if np.isfinite(gap_even) and gap_even > tol:
        warn(f"{ds.path.name}: gap even mismatch {gap_even:.3e} exceeds tolerance {tol:.1e}")
    if np.isfinite(mx_even) and mx_even > tol:
        warn(f"{ds.path.name}: mx even mismatch {mx_even:.3e} exceeds tolerance {tol:.1e}")


# ---------------------------------------------------------------------------
# Longitudinal susceptibility
# ---------------------------------------------------------------------------
def _unique_by_tolerance(x: np.ndarray, arrays: list[np.ndarray], *, tol: float) -> tuple[np.ndarray, list[np.ndarray], int]:
    order = np.argsort(x)
    x_sorted = np.asarray(x, dtype=float)[order]
    arrays_sorted = [np.asarray(arr, dtype=float)[order] for arr in arrays]
    keep = np.ones(x_sorted.size, dtype=bool)
    duplicates = 0
    last = np.nan
    for i, value in enumerate(x_sorted):
        if np.isfinite(last) and abs(value - last) <= tol:
            keep[i] = False
            duplicates += 1
            continue
        last = value
    return x_sorted[keep], [arr[keep] for arr in arrays_sorted], duplicates


def compute_chi_z_cqt(dataset: HFieldDataset) -> dict:
    if dataset.mode != "cqt":
        raise ValueError("compute_chi_z_cqt expects a CQT dataset")
    data = dataset.data[np.argsort(dataset.data[:, HCOL["h"]])]
    h = data[:, HCOL["h"]]
    scale_x = data[:, HCOL["scale_x"]]
    mz = data[:, HCOL["mz"]]
    method = data[:, HCOL["method_code"]]
    h, (scale_x, mz, method), duplicates = _unique_by_tolerance(
        h, [scale_x, mz, method], tol=1e-14
    )
    if duplicates:
        warn(f"{dataset.path.name}: removed {duplicates} duplicate h rows within 1e-14 for chi_z")

    mz_sym = np.array(mz, copy=True)
    missing_opposites = 0
    pairs = _paired_values(h, mz, tol=1e-13)
    oddness = float(max(abs(yp + ym) for _x, yp, ym in pairs)) if pairs else np.nan
    for i, hi in enumerate(h):
        j = int(np.argmin(np.abs(h + hi))) if h.size else -1
        if j >= 0 and abs(h[j] + hi) <= 1e-13:
            mz_sym[i] = 0.5 * (mz[i] - mz[j])
        else:
            mz_sym[i] = mz[i]
            missing_opposites += 1
    if missing_opposites:
        warn(
            f"{dataset.path.name}: {missing_opposites} h points lack an opposite; "
            "chi_z uses raw mz there"
        )

    if h.size < 2:
        warn(f"{dataset.path.name}: too few h points for chi_z interpolation")
        chi = np.full_like(h, np.nan, dtype=float)
        interp_method = "none"
    else:
        try:
            interpolant = PchipInterpolator(h, mz_sym, extrapolate=False)
            chi = np.asarray(interpolant.derivative()(h), dtype=float)
            interp_method = "PchipInterpolator"
        except (ValueError, FloatingPointError) as exc:
            warn(f"{dataset.path.name}: PCHIP failed ({exc}); trying CubicSpline")
            try:
                interpolant = CubicSpline(h, mz_sym)
                chi = np.asarray(interpolant(h, 1), dtype=float)
                interp_method = "CubicSpline"
            except (ValueError, FloatingPointError) as exc2:
                warn(f"{dataset.path.name}: CubicSpline failed ({exc2})")
                chi = np.full_like(h, np.nan, dtype=float)
                interp_method = "failed"

    finite = np.isfinite(chi)
    non_positive = int(np.count_nonzero(finite & (chi <= 0.0)))
    positive_chi = chi[finite & (chi > 0.0)]
    chi_min = float(np.min(positive_chi)) if positive_chi.size else np.nan
    chi_max = float(np.max(positive_chi)) if positive_chi.size else np.nan
    tol = 1e-10 if dataset.method_code == 0 else 1e-8
    print(
        f"  chi_z {dataset.bc_label} L={dataset.L:2d}: method={interp_method} "
        f"points={h.size} max_oddness={oddness:.3e} "
        f"non_positive={non_positive} chi_pos_range=[{chi_min:.6e}, {chi_max:.6e}]"
    )
    if np.isfinite(oddness) and oddness > tol:
        warn(f"{dataset.path.name}: chi_z mz oddness {oddness:.3e} exceeds tolerance {tol:.1e}")
    if non_positive:
        warn(f"{dataset.path.name}: chi_z has {non_positive} non-positive finite values")

    return dict(
        dataset=dataset,
        h=h,
        scale_x=scale_x,
        mz=mz,
        mz_sym=mz_sym,
        chi_z=chi,
        chi_positive=(np.isfinite(chi) & (chi > 0.0)).astype(int),
        method_code=method.astype(int),
        interpolation=interp_method,
        max_oddness=oddness,
        non_positive=non_positive,
    )


def build_chi_tables(cqt_data: dict[str, list[HFieldDataset]], constants: dict) -> dict[str, list[dict]]:
    chi_by_bc: dict[str, list[dict]] = {"PBC": [], "OBC": []}
    for bc in ("PBC", "OBC"):
        for ds in cqt_data.get(bc, []):
            chi_by_bc[bc].append(compute_chi_z_cqt(ds))
    write_chi_tables(chi_by_bc, constants)
    return chi_by_bc


def write_chi_tables(chi_by_bc: dict[str, list[dict]], constants: dict) -> None:
    for bc in ("PBC", "OBC"):
        rows = chi_by_bc.get(bc, [])
        if not rows:
            continue
        path = PROCESSED_DIR / f"cqt_chi_{bc.lower()}.dat"
        with path.open("w", encoding="utf-8") as fp:
            fp.write("# Longitudinal susceptibility for h-field CQT data\n")
            fp.write("# Definition: chi_z(h,L) = d m_z(h,L) / d h\n")
            fp.write("# Interpolation: odd-symmetrized m_z with PchipInterpolator derivative; CubicSpline fallback\n")
            fp.write(
                "# Exponents: "
                f"g_pc={float(constants[bc]['g_pc']):.12f} "
                f"beta_over_nu={float(constants[bc]['beta_over_nu']):.12e} "
                f"gamma_over_nu={float(constants[bc]['gamma_over_nu']):.12e} "
                f"y_h={float(constants[bc]['y_h']):.12e} "
                f"delta={float(constants[bc]['delta']):.12e}\n"
            )
            fp.write("# Sources:\n")
            for item in rows:
                ds = item["dataset"]
                fp.write(f"#   L={ds.L} {ds.path.relative_to(PROJECT_ROOT)} interpolation={item['interpolation']}\n")
            fp.write("# Columns: L h scale_x mz mz_sym chi_z chi_positive_flag method_code\n")
            for item in rows:
                ds = item["dataset"]
                for values in zip(
                    item["h"],
                    item["scale_x"],
                    item["mz"],
                    item["mz_sym"],
                    item["chi_z"],
                    item["chi_positive"],
                    item["method_code"],
                ):
                    h, scale_x, mz, mz_sym, chi_z, flag, method_code = values
                    fp.write(
                        f"{ds.L:4d} {h:.16e} {scale_x:.16e} {mz:.16e} "
                        f"{mz_sym:.16e} {chi_z:.16e} {int(flag):1d} {int(method_code):1d}\n"
                    )
        SAVED_TABLES.append(path)
        print(f"  [OK] {path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def _datasets_for_panel(cqt_or_foqt: dict, bc: str) -> list[HFieldDataset]:
    return list(cqt_or_foqt.get(bc, []))


def _plot_dataset_line(ax: plt.Axes, ds: HFieldDataset,
                       x: np.ndarray, y: np.ndarray,
                       *, lw: float = 1.35, label: str | None = None) -> bool:
    m = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(m) < 1:
        return False
    colors = colors_for_bc(ds.bc_label)
    ax.plot(
        x[m], y[m],
        color=colors.get(ds.L, "gray"),
        ls=_line_style(ds),
        alpha=_line_alpha(ds),
        lw=lw + LINEWIDTH_OFFSET,
        label=_line_label(ds) if label is None else label,
    )
    return True


def _panel_setup(ax: plt.Axes, bc_label: str, *, xlabel: str, ylabel: str) -> None:
    ax.set_title(bc_label, loc="right", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    apply_ticks(ax)
    apply_grid(ax)


def _reference_powerlaw(ax: plt.Axes, x: np.ndarray, y: np.ndarray,
                        exponent: float, *, color, label: str) -> None:
    m = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    if np.count_nonzero(m) < 2 or not np.isfinite(exponent):
        return
    x_use = x[m]
    y_use = y[m]
    order = np.argsort(x_use)
    x_use = x_use[order]
    y_use = y_use[order]
    anchor_idx = len(x_use) // 2
    x0 = float(x_use[anchor_idx])
    y0 = float(y_use[anchor_idx])
    xlo = float(np.min(x_use))
    xhi = float(np.max(x_use))
    if xhi <= xlo:
        return
    xs = np.geomspace(xlo, xhi, 200)
    ys = y0 * (xs / x0) ** exponent
    ax.plot(xs, ys, color=color, ls=":", lw=1.4, label=label)


def _kappa_reference_grid(kappa_values: list[np.ndarray]) -> np.ndarray:
    kmin, kmax = finite_minmax(kappa_values)
    if not (np.isfinite(kmin) and np.isfinite(kmax)):
        return np.asarray([], dtype=float)
    if kmax > kmin:
        pad = 0.03 * (kmax - kmin)
    else:
        pad = 0.03 * max(abs(kmin), 1.0)
    return np.linspace(kmin - pad, kmax + pad, 600)


def _add_foqt_raw_zoom_inset(
    ax: plt.Axes,
    datasets: list[HFieldDataset],
    *,
    y_shift: float = 0.0,
    loc: str = "center right",
    width: str = "32%",
    height: str = "32%",
    bbox_to_anchor=None,
    bbox_transform=None,
) -> None:
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    inset_kwargs = dict(
        width=width,
        height=height,
        loc=loc,
        borderpad=1.32,
    )
    if bbox_to_anchor is not None:
        inset_kwargs.update(
            bbox_to_anchor=bbox_to_anchor,
            bbox_transform=bbox_transform or ax.transAxes,
        )
    elif y_shift != 0.0:
        inset_kwargs.update(
            bbox_to_anchor=(0.0, y_shift, 1.0, 1.0),
            bbox_transform=ax.transAxes,
        )
    axins = inset_axes(ax, **inset_kwargs)
    has_data = False
    for ds in datasets:
        plotted = _plot_dataset_line(
            axins,
            ds,
            ds.data[:, HCOL["h"]],
            ds.data[:, HCOL["mz"]],
            lw=1.0,
            label="_nolegend_",
        )
        has_data = has_data or plotted

    if not has_data:
        axins.set_visible(False)
        return

    axins.axvline(0.0, color="gray", ls=":", lw=0.9, alpha=0.75)
    axins.axhline(0.0, color="gray", ls=":", lw=0.9, alpha=0.75)
    axins.set(xlim=(-2.0e-4, 2.0e-4), ylim=(-2.5e-3, 2.5e-3))
    axins.tick_params(direction="in", which="both", top=True, right=True,
                      labelsize=INSET_TICK_FONTSIZE)
    apply_grid(axins)


def _add_cqt_delta_inset(ax: plt.Axes, datasets: list[HFieldDataset],
                         constants: dict, bc: str) -> bool:
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib.ticker import FixedFormatter, FixedLocator, LogLocator, NullFormatter

    if not datasets:
        warn(f"CQT {bc}: delta inset skipped because no datasets are available")
        return False

    try:
        delta = float(constants[bc]["delta"])
    except (KeyError, TypeError, ValueError) as exc:
        warn(f"CQT {bc}: delta inset skipped because delta is unavailable ({exc})")
        return False
    if not np.isfinite(delta) or delta == 0.0:
        warn(f"CQT {bc}: delta inset skipped because delta={delta!r} is invalid")
        return False

    primary_range = (1.0e-2, 5.0e-2)
    min_points = 2

    def usable_arrays(ds: HFieldDataset, *, primary: bool) -> tuple[np.ndarray, np.ndarray]:
        h = np.asarray(ds.data[:, HCOL["h"]], dtype=float)
        mz_abs = np.abs(np.asarray(ds.data[:, HCOL["mz"]], dtype=float))
        mask = np.isfinite(h) & np.isfinite(mz_abs) & (h > 0.0) & (mz_abs > 0.0)
        if primary:
            mask &= (h >= primary_range[0]) & (h <= primary_range[1])
        h_use = h[mask]
        y_use = mz_abs[mask]
        if h_use.size:
            order = np.argsort(h_use)
            h_use = h_use[order]
            y_use = y_use[order]
        return h_use, y_use

    primary_counts = sum(usable_arrays(ds, primary=True)[0].size for ds in datasets)
    use_primary = primary_counts >= min_points
    if use_primary:
        largest = max(datasets, key=lambda item: item.L)
        if usable_arrays(largest, primary=True)[0].size < min_points:
            use_primary = False

    all_h: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    per_dataset: list[tuple[HFieldDataset, np.ndarray, np.ndarray]] = []
    for ds in datasets:
        h_use, y_use = usable_arrays(ds, primary=use_primary)
        if h_use.size:
            per_dataset.append((ds, h_use, y_use))
            all_h.append(h_use)
            all_y.append(y_use)

    if not per_dataset:
        warn(f"CQT {bc}: delta inset skipped because no positive h, |m_z| data are available")
        return False

    anchor_item: tuple[HFieldDataset, np.ndarray, np.ndarray] | None = None
    for item in sorted(per_dataset, key=lambda value: value[0].L, reverse=True):
        if item[1].size >= min_points:
            anchor_item = item
            break
    if anchor_item is None:
        warn(f"CQT {bc}: delta inset skipped because fewer than two useful points are available")
        return False

    h_all = np.concatenate(all_h)
    y_all = np.concatenate(all_y)
    positive = np.isfinite(h_all) & np.isfinite(y_all) & (h_all > 0.0) & (y_all > 0.0)
    if np.count_nonzero(positive) < min_points:
        warn(f"CQT {bc}: delta inset skipped because positive data are insufficient")
        return False

    axins = inset_axes(
        ax,
        width="36%",
        height="36%",
        loc="upper left",
        bbox_to_anchor=(0.045, 0.0, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=1.15,
    )
    axins.patch.set_facecolor("white")
    axins.patch.set_alpha(0.88)
    axins.patch.set_edgecolor("0.55")
    axins.patch.set_linewidth(0.8)
    for spine in axins.spines.values():
        spine.set_color("0.55")
        spine.set_linewidth(0.8)

    for ds, h_use, y_use in per_dataset:
        _plot_dataset_line(axins, ds, h_use, y_use, lw=0.95, label="_nolegend_")

    _ds_anchor, h_anchor_values, y_anchor_values = anchor_item
    log_center = 0.5 * (np.log(h_anchor_values[0]) + np.log(h_anchor_values[-1]))
    anchor_idx = int(np.argmin(np.abs(np.log(h_anchor_values) - log_center)))
    h_anchor = float(h_anchor_values[anchor_idx])
    y_anchor = float(y_anchor_values[anchor_idx])
    exponent = 1.0 / delta
    if h_anchor <= 0.0 or y_anchor <= 0.0 or not np.isfinite(exponent):
        axins.set_visible(False)
        warn(f"CQT {bc}: delta inset skipped because the theory anchor is invalid")
        return False

    h_ref_min = float(np.min(h_all[positive]))
    h_ref_max = float(np.max(h_all[positive]))
    if not (h_ref_max > h_ref_min > 0.0):
        axins.set_visible(False)
        warn(f"CQT {bc}: delta inset skipped because h range is degenerate")
        return False
    xlim = primary_range if use_primary else (h_ref_min, h_ref_max)
    h_ref = np.geomspace(xlim[0], xlim[1], 200)
    amplitude = y_anchor / (h_anchor ** exponent)
    y_ref = amplitude * (h_ref ** exponent)
    axins.plot(
        h_ref,
        y_ref,
        color="lime" if bc.upper() == "PBC" else "red",
        ls="--",
        lw=1.1,
        label=r"$h^{1/\delta}$",
        zorder=4,
    )

    axins.set_xscale("log")
    axins.set_yscale("log")
    axins.set_xlim(xlim)
    if use_primary:
        axins.xaxis.set_major_locator(FixedLocator([primary_range[0], primary_range[1]]))
        axins.xaxis.set_major_formatter(
            FixedFormatter([r"$10^{-2}$", r"$5{\times}10^{-2}$"])
        )
        axins.xaxis.set_minor_formatter(NullFormatter())
    if bc.upper() == "PBC":
        axins.yaxis.set_major_locator(FixedLocator([0.2, 0.4, 0.6, 0.8]))
        axins.yaxis.set_major_formatter(FixedFormatter(["0.2", "0.4", "0.6", "0.8"]))
    else:
        axins.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=4))
    axins.yaxis.set_minor_formatter(NullFormatter())
    axins.tick_params(direction="in", which="both", top=True, right=True,
                      labelsize=TICK_FONTSIZE - 1, pad=1)
    axins.grid(True, which="both", axis="both",
               color="0.75", alpha=0.45, linestyle=":", linewidth=0.6, zorder=0)
    legend = axins.legend(loc="lower right", frameon=True,
                          fontsize=LEGEND_FONTSIZE,
                          handlelength=1.6, borderpad=0.25,
                          labelspacing=0.2)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.80)
    legend.get_frame().set_edgecolor("none")
    return True


import contextlib

@contextlib.contextmanager
def _font_bump(delta: int = 2, lw_delta: float = 0.4):
    global AXIS_LABEL_FONTSIZE, TICK_FONTSIZE, TITLE_FONTSIZE, LEGEND_FONTSIZE, TEXT_FONTSIZE, LINEWIDTH_OFFSET
    old = (AXIS_LABEL_FONTSIZE, TICK_FONTSIZE, TITLE_FONTSIZE, LEGEND_FONTSIZE, TEXT_FONTSIZE, LINEWIDTH_OFFSET)
    AXIS_LABEL_FONTSIZE += delta
    TICK_FONTSIZE += delta
    TITLE_FONTSIZE += delta
    LEGEND_FONTSIZE += delta
    TEXT_FONTSIZE += delta
    LINEWIDTH_OFFSET += lw_delta

    import matplotlib as mpl
    old_rc = {
        "axes.linewidth": mpl.rcParams["axes.linewidth"],
        "xtick.major.width": mpl.rcParams["xtick.major.width"],
        "ytick.major.width": mpl.rcParams["ytick.major.width"],
        "xtick.minor.width": mpl.rcParams["xtick.minor.width"],
        "ytick.minor.width": mpl.rcParams["ytick.minor.width"],
    }
    mpl.rcParams["axes.linewidth"] += lw_delta
    mpl.rcParams["xtick.major.width"] += lw_delta
    mpl.rcParams["ytick.major.width"] += lw_delta
    mpl.rcParams["xtick.minor.width"] += lw_delta
    mpl.rcParams["ytick.minor.width"] += lw_delta

    try:
        yield
    finally:
        AXIS_LABEL_FONTSIZE, TICK_FONTSIZE, TITLE_FONTSIZE, LEGEND_FONTSIZE, TEXT_FONTSIZE, LINEWIDTH_OFFSET = old
        mpl.rcParams.update(old_rc)


# ---------------------------------------------------------------------------
# CQT plots
# ---------------------------------------------------------------------------
def plot_cqt_order_parameter(cqt_data: dict[str, list[HFieldDataset]], constants: dict,
                             *, min_L: int) -> bool:
    rc_params = {}
    with _font_bump(3, 0.4), plt.rc_context(rc_params):
        fig, axes2d = plt.subplots(2, 2, figsize=(14.0, 11.0), constrained_layout=True)
        axes = axes2d.flatten()
        fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.03, wspace=0.02, hspace=0.02)
        any_data = False
        plotted_L_methods: dict[int, set[int]] = {}
        plotted_methods: set[int] = set()

        panel_specs = [
            ("PBC", "raw", axes[0], r"$h$", r"$m_z$"),
            ("OBC", "raw", axes[1], r"$h$", ""),
            ("PBC", "collapse", axes[2], r"$hL^{y_h}$", r"$m_z L^{\beta/\nu}$"),
            ("OBC", "collapse", axes[3], r"$hL^{y_h}$", ""),
        ]
        panel_map = {(bc, kind): (ax, xlabel, ylabel) for bc, kind, ax, xlabel, ylabel in panel_specs}

        for bc in ("PBC", "OBC"):
            datasets = _datasets_for_panel(cqt_data, bc)
            complete_count = sum(1 for ds in datasets if ds.complete)
            if not datasets:
                warn(f"CQT {bc}: no datasets for order parameter")
            elif complete_count == 0:
                warn(f"CQT {bc}: no complete datasets for order parameter; plotting partial only")

            ax_raw, raw_xlabel, raw_ylabel = panel_map[(bc, "raw")]
            ax_collapse, collapse_xlabel, collapse_ylabel = panel_map[(bc, "collapse")]

            if not datasets:
                _panel_setup(ax_raw, bc, xlabel=raw_xlabel, ylabel=raw_ylabel)
                _panel_setup(ax_collapse, bc, xlabel=collapse_xlabel, ylabel=collapse_ylabel)
                _no_data(ax_raw, "No data")
                _no_data(ax_collapse, "No data")
                ax_raw.set_xlim((-0.25, 0.25))
                ax_collapse.set_xlim((-5.0, 5.0))
                continue

            has_positive = False
            has_nonpositive = False
            for ds in datasets:
                h = ds.data[:, HCOL["h"]]
                mz = ds.data[:, HCOL["mz"]]
                m = np.isfinite(h) & np.isfinite(mz)
                has_nonpositive = has_nonpositive or bool(np.any(m & (mz == 0.0)))
                mpos = m & (np.abs(mz) > 0.0)
                if np.count_nonzero(mpos) < 1:
                    continue
                if _plot_dataset_line(ax_raw, ds, h[mpos], mz[mpos]):
                    _register_external_legend_dataset(ds, plotted_L_methods, plotted_methods)
                has_positive = True

            ax_raw.axvline(0.0, color="gray", ls=":", lw=1.1 + LINEWIDTH_OFFSET, alpha=0.75)
            _panel_setup(ax_raw, bc, xlabel=raw_xlabel, ylabel=raw_ylabel)
            if has_positive:
                ax_raw.set_ylim((-1.0, 1.0))
            else:
                _no_data(ax_raw, "No positive m_z")
            ax_raw.set_xlim((-0.25, 0.25))
            _add_cqt_delta_inset(ax_raw, datasets, constants, bc)

            if has_nonpositive:
                warn(f"CQT {bc}: zero m_z values omitted from order-parameter plot")

            collapse_datasets = [ds for ds in datasets if ds.L >= min_L]
            if not collapse_datasets and datasets:
                warn(f"CQT {bc}: no datasets with L >= {min_L} for order collapse")
            beta_over_nu = float(constants[bc]["beta_over_nu"])
            has_positive_collapse = False
            has_nonpositive_collapse = False
            for ds in collapse_datasets:
                x = ds.data[:, HCOL["scale_x"]]
                y = ds.data[:, HCOL["mz"]] * (float(ds.L) ** beta_over_nu)
                m = np.isfinite(x) & np.isfinite(y)
                has_nonpositive_collapse = has_nonpositive_collapse or bool(np.any(m & (y == 0.0)))
                mpos = m & (np.abs(y) > 0.0)
                if np.count_nonzero(mpos) < 1:
                    continue
                if _plot_dataset_line(ax_collapse, ds, x[mpos], y[mpos]):
                    _register_external_legend_dataset(ds, plotted_L_methods, plotted_methods)
                has_positive_collapse = True

            ax_collapse.axvline(0.0, color="gray", ls=":", lw=1.1 + LINEWIDTH_OFFSET, alpha=0.75)
            _panel_setup(ax_collapse, bc, xlabel=collapse_xlabel, ylabel=collapse_ylabel)
            if has_positive_collapse:
                ax_collapse.set_ylim((-1.5, 1.5))
            else:
                _no_data(ax_collapse, "No positive m_z")
            ax_collapse.set_xlim((-5.0, 5.0))

            if has_nonpositive_collapse:
                warn(f"CQT {bc}: zero m_z values omitted from collapse plot")

            any_data = any_data or has_positive or has_positive_collapse

        axes[1].tick_params(labelleft=False)
        axes[3].tick_params(labelleft=False)

        if not any_data:
            plt.close(fig)
            warn("CQT order parameter: no data available; figure skipped")
            record_figure("plots/hfield/cqt_order_parameter.pdf", False, "no data")
            return False

        _external_size_method_legend(fig, plotted_L_methods, plotted_methods, y_anchor=-0.08)
        save_fig(fig, "cqt_order_parameter.pdf")
        record_figure("plots/hfield/cqt_order_parameter.pdf", True)
        return True


def plot_cqt_susceptibility(chi_by_bc: dict[str, list[dict]], constants: dict,
                            *, min_L: int) -> bool:
    rc_params = {}
    with _font_bump(3, 0.4), plt.rc_context(rc_params):
        fig, axes2d = plt.subplots(2, 2, figsize=(14.0, 11.0), constrained_layout=True)
        axes = axes2d.flatten()
        fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.03, wspace=0.02, hspace=0.02)
        any_data = False
        plotted_L_methods: dict[int, set[int]] = {}
        plotted_methods: set[int] = set()
        pbc_raw_ylim: tuple[float, float] | None = None
        pbc_collapse_ylim: tuple[float, float] | None = None

        panel_specs = [
            ("PBC", "raw", axes[0], r"$h$", r"$\chi_z$"),
            ("OBC", "raw", axes[1], r"$h$", ""),
            ("PBC", "collapse", axes[2], r"$hL^{y_H}$", r"$\chi_z L^{-\gamma/\nu}$"),
            ("OBC", "collapse", axes[3], r"$hL^{y_H}$", ""),
        ]
        panel_map = {(bc, kind): (ax, xlabel, ylabel) for bc, kind, ax, xlabel, ylabel in panel_specs}

        for bc in ("PBC", "OBC"):
            items = list(chi_by_bc.get(bc, []))
            ax_raw, raw_xlabel, raw_ylabel = panel_map[(bc, "raw")]
            ax_collapse, collapse_xlabel, collapse_ylabel = panel_map[(bc, "collapse")]

            if not items:
                warn(f"CQT {bc}: no chi_z data for susceptibility")
                _panel_setup(ax_raw, bc, xlabel=raw_xlabel, ylabel=raw_ylabel)
                _panel_setup(ax_collapse, bc, xlabel=collapse_xlabel, ylabel=collapse_ylabel)
                _no_data(ax_raw, "No data")
                _no_data(ax_collapse, "No data")
                ax_raw.set_xlim((-0.25, 0.25))
                ax_collapse.set_xlim((-5.0, 5.0))
                continue

            has_positive = False
            has_nonpositive = False
            for item in items:
                ds = item["dataset"]
                h = item["h"]
                chi = item["chi_z"]
                m = np.isfinite(h) & np.isfinite(chi)
                has_nonpositive = has_nonpositive or bool(np.any(m & (chi <= 0.0)))
                mpos = m & (chi > 0.0)
                if np.count_nonzero(mpos) < 1:
                    continue
                if _plot_dataset_line(ax_raw, ds, h[mpos], chi[mpos]):
                    _register_external_legend_dataset(ds, plotted_L_methods, plotted_methods)
                has_positive = True
            ax_raw.axvline(0.0, color="gray", ls=":", lw=1.1 + LINEWIDTH_OFFSET, alpha=0.75)
            _panel_setup(ax_raw, bc, xlabel=raw_xlabel, ylabel=raw_ylabel)
            if has_positive:
                ax_raw.set_yscale("log")
                autoscale_independent(ax_raw)

                y_mins = []
                for item in items:
                    m = (item["h"] >= -0.1) & (item["h"] <= 0.1) & (item["chi_z"] > 0.0)
                    if np.any(m):
                        y_mins.append(np.min(item["chi_z"][m]))
                if y_mins:
                    ax_raw.set_ylim(bottom=min(y_mins) * 0.8)

                if bc == "PBC":
                    pbc_raw_ylim = ax_raw.get_ylim()
            else:
                _no_data(ax_raw, "No positive chi_z")
            ax_raw.set_xlim((-0.1, 0.1))

            if has_nonpositive:
                warn(f"CQT {bc}: non-positive chi_z values omitted from susceptibility plot")

            items_collapse = [item for item in items if item["dataset"].L >= min_L]
            if not items_collapse and items:
                warn(f"CQT {bc}: no chi_z datasets with L >= {min_L} for collapse")
            gamma_over_nu = float(constants[bc]["gamma_over_nu"])
            has_positive_collapse = False
            has_nonpositive_collapse = False
            for item in items_collapse:
                ds = item["dataset"]
                x = item["scale_x"]
                y = item["chi_z"] * (float(ds.L) ** (-gamma_over_nu))
                m = np.isfinite(x) & np.isfinite(y)
                has_nonpositive_collapse = has_nonpositive_collapse or bool(np.any(m & (y <= 0.0)))
                mpos = m & (y > 0.0)
                if np.count_nonzero(mpos) < 1:
                    continue
                if _plot_dataset_line(ax_collapse, ds, x[mpos], y[mpos]):
                    _register_external_legend_dataset(ds, plotted_L_methods, plotted_methods)
                has_positive_collapse = True
            ax_collapse.axvline(0.0, color="gray", ls=":", lw=1.1 + LINEWIDTH_OFFSET, alpha=0.75)
            _panel_setup(ax_collapse, bc, xlabel=collapse_xlabel, ylabel=collapse_ylabel)
            if has_positive_collapse:
                ax_collapse.set_yscale("log")
                autoscale_independent(ax_collapse)
                ax_collapse.set_ylim(bottom=1e-2)
                if bc == "PBC":
                    pbc_collapse_ylim = ax_collapse.get_ylim()
            else:
                _no_data(ax_collapse, "No positive chi_z")
            ax_collapse.set_xlim((-5.0, 5.0))

            if has_nonpositive_collapse:
                warn(f"CQT {bc}: non-positive chi_z values omitted from collapse plot")

            any_data = any_data or has_positive or has_positive_collapse

        if pbc_raw_ylim is not None:
            axes[1].set_ylim(pbc_raw_ylim)
            axes[1].tick_params(labelleft=False)
        if pbc_collapse_ylim is not None:
            axes[3].set_ylim(pbc_collapse_ylim)
            axes[3].tick_params(labelleft=False)

        if not any_data:
            plt.close(fig)
            warn("CQT susceptibility: no data available; figure skipped")
            record_figure("plots/hfield/cqt_susceptibility.pdf", False, "no data")
            return False

        for ax_idx in (0, 1):
            ax = axes[ax_idx]
            ticks = [-0.075, -0.025, 0.0, 0.025, 0.075]
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t:g}" for t in ticks])

        _external_size_method_legend(fig, plotted_L_methods, plotted_methods, y_anchor=-0.08)
        save_fig(fig, "cqt_susceptibility.pdf")
        record_figure("plots/hfield/cqt_susceptibility.pdf", True)
        return True


# ---------------------------------------------------------------------------
# FOQT plots
# ---------------------------------------------------------------------------
def _foqt_datasets(foqt_data: dict, target_g: float, bc: str) -> list[HFieldDataset]:
    return list(foqt_data.get(float(target_g), {}).get(bc, []))


def _m0_for_dataset(ds: HFieldDataset, target_g: float) -> float:
    if np.isfinite(ds.m0) and ds.m0 > 0.0:
        return float(ds.m0)
    if abs(target_g) < 1.0:
        fallback = float((1.0 - target_g ** 2) ** 0.125)
        warn(f"{ds.path.name}: m0 missing; using (1-g^2)^(1/8)={fallback:.12e}")
        return fallback
    warn(f"{ds.path.name}: m0 missing and g is outside ordered phase")
    return np.nan


def _delta0_for_dataset(ds: HFieldDataset) -> float:
    if np.isfinite(ds.delta0_h0) and ds.delta0_h0 > 0.0:
        return float(ds.delta0_h0)
    col = ds.data[:, HCOL["delta0_h0"]]
    col = col[np.isfinite(col) & (col > 0.0)]
    if col.size:
        fallback = float(np.median(col))
        warn(f"{ds.path.name}: delta0_h0 header missing; using data median={fallback:.12e}")
        return fallback
    warn(f"{ds.path.name}: delta0_h0 missing")
    return np.nan


def _get_mz_ext_data(ds: HFieldDataset) -> tuple[np.ndarray, np.ndarray] | None:
    # Optional local extension data are read only when --use-local-extensions is passed; extension generators are intentionally not part of the public repository.
    if not ds.path.name.startswith("hfield_"):
        return None
    ext_name = ds.path.name.replace("hfield_", "mzext_")
    ext_path = ds.path.parents[1] / "mz_ext" / ext_name
    if not ext_path.exists():
        return None
    try:
        data = np.loadtxt(ext_path, comments="#")
        if data.ndim == 2 and data.shape[1] >= 3:
            return data[:, 0], data[:, 2]
    except Exception as exc:
        warn(f"Failed to load mz_ext for {ds.path.name}: {exc}")
    return None


def plot_combined_foqt_scaling(
    foqt_data: dict,
    foqt_g_list: list[float],
    *,
    min_L: int,
    use_local_extensions: bool,
) -> bool:
    fig = plt.figure(
        figsize=(FOQT_REPORT_FIGURE_WIDTH_IN, FOQT_REPORT_FIGURE_HEIGHT_IN),
        constrained_layout=False,
    )
    outer_gs = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[4.0, FOQT_LEGEND_HEIGHT_RATIO],
        hspace=FOQT_LEGEND_HSPACE,
    )
    plot_gs = outer_gs[0].subgridspec(
        nrows=4,
        ncols=6,
        hspace=FOQT_PANEL_HSPACE,
        wspace=FOQT_PANEL_WSPACE,
    )
    axes = np.empty((4, 3), dtype=object)
    for row_idx in range(4):
        for col_idx in range(3):
            axes[row_idx, col_idx] = fig.add_subplot(
                plot_gs[row_idx, 2 * col_idx:2 * col_idx + 2]
            )

    any_data = False
    plotted_L_methods: dict[int, set[int]] = {}
    plotted_methods: set[int] = set()

    for row_idx in range(4):
        if row_idx < 2:
            target_g = foqt_g_list[0]
            bc = "PBC" if row_idx == 0 else "OBC"
        else:
            target_g = foqt_g_list[1]
            bc = "PBC" if row_idx == 2 else "OBC"

        datasets = _foqt_datasets(foqt_data, target_g, bc)
        if not datasets:
            warn(f"FOQT {bc} g={_target_g_label(target_g)}: no datasets for scaling")

        is_last_row = (row_idx == 3)
        xlabel_raw = r"$h$" if is_last_row else ""
        xlabel_kappa = r"$\kappa$" if is_last_row else ""

        ax_raw = axes[row_idx, 0]
        ax_mz = axes[row_idx, 1]
        ax_gap = axes[row_idx, 2]

        if datasets:
            any_data = True
            for ds in datasets:
                x_mz = ds.data[:, HCOL["h"]]
                y_mz = ds.data[:, HCOL["mz"]]
                ext_data = _get_mz_ext_data(ds) if use_local_extensions else None
                if ext_data is not None:
                    x_mz, y_mz = ext_data
                if _plot_dataset_line(ax_raw, ds, x_mz, y_mz):
                    _register_external_legend_dataset(ds, plotted_L_methods, plotted_methods)
            ax_raw.axvline(0.0, color="gray", ls=":", lw=1.1, alpha=0.75)
            ax_raw.axhline(0.0, color="gray", ls=":", lw=1.1, alpha=0.75)
            _panel_setup(ax_raw, bc, xlabel=xlabel_raw, ylabel=r"$m_z$")
            ax_raw.set_title(f"{bc} (g={target_g})", loc="right", fontsize=TITLE_FONTSIZE)
            autoscale_independent(ax_raw)

            collapse = [ds for ds in datasets if ds.L >= min_L]
            if not collapse:
                warn(f"FOQT {bc} g={_target_g_label(target_g)}: no datasets with L >= {min_L}")

            plotted_mz: list[HFieldDataset] = []
            kappa_mz_values: list[np.ndarray] = []
            y_mz_values: list[np.ndarray] = []
            for ds in collapse:
                m0 = _m0_for_dataset(ds, target_g)
                if not np.isfinite(m0) or m0 == 0.0:
                    continue
                x = ds.data[:, HCOL["kappa"]]
                y = ds.data[:, HCOL["mz"]] / m0
                ext_data = _get_mz_ext_data(ds) if use_local_extensions else None
                if ext_data is not None:
                    h_ext, mz_ext = ext_data
                    delta0 = _delta0_for_dataset(ds)
                    if np.isfinite(delta0) and delta0 > 0.0:
                        x = 2.0 * m0 * h_ext * ds.L / delta0
                        y = mz_ext / m0
                if _plot_dataset_line(ax_mz, ds, x, y):
                    _register_external_legend_dataset(ds, plotted_L_methods, plotted_methods)
                    plotted_mz.append(ds)
                    kappa_mz_values.append(x)
                    y_mz_values.append(y)
            if plotted_mz:
                theory_handles = []
                k_ref = _kappa_reference_grid(kappa_mz_values)
                if k_ref.size:
                    mz_ref = k_ref / np.sqrt(1.0 + k_ref ** 2)
                    theory_line, = ax_mz.plot(
                        k_ref,
                        mz_ref,
                        color="0.35",
                        ls="--",
                        lw=1.6,
                        label=r"$\kappa/\sqrt{1+\kappa^2}$",
                    )
                    theory_handles.append(theory_line)
                ax_mz.axvline(0.0, color="gray", ls=":", lw=1.1, alpha=0.75)
                ax_mz.axhline(0.0, color="gray", ls=":", lw=1.1, alpha=0.75)
                _panel_setup(ax_mz, bc, xlabel=xlabel_kappa, ylabel=r"$m_z/m_0$")
                ax_mz.set_title(f"{bc} (g={target_g})", loc="right", fontsize=TITLE_FONTSIZE)
                _theory_relation_legend(ax_mz, theory_handles, loc="lower right", fontsize=LEGEND_FONTSIZE - 2)
                if bc == "OBC":
                    kmin, kmax = finite_minmax(kappa_mz_values)
                    ymin, ymax = finite_minmax(y_mz_values)
                    if np.isfinite(kmin) and np.isfinite(kmax):
                        span = kmax - kmin
                        if span > 0:
                            ax_mz.set_xlim(kmin - 0.05 * span, kmax + 0.05 * span)
                    if np.isfinite(ymin) and np.isfinite(ymax):
                        span = ymax - ymin
                        if span > 0:
                            ax_mz.set_ylim(ymin - 0.05 * span, ymax + 0.05 * span)
                else:
                    autoscale_independent(ax_mz)
            else:
                _panel_setup(ax_mz, bc, xlabel=xlabel_kappa, ylabel=r"$m_z/m_0$")
                ax_mz.set_title(f"{bc} (g={target_g})", loc="right", fontsize=TITLE_FONTSIZE)
                _no_data(ax_mz, "No valid data for collapse")

            plotted_gap: list[HFieldDataset] = []
            kappa_gap_values: list[np.ndarray] = []
            for ds in collapse:
                m0 = _m0_for_dataset(ds, target_g)
                if not np.isfinite(m0) or m0 == 0.0:
                    continue
                delta0 = _delta0_for_dataset(ds)
                if not np.isfinite(delta0) or delta0 == 0.0:
                    continue
                x = ds.data[:, HCOL["kappa"]]
                y = ds.data[:, HCOL["delta_h"]] / delta0
                if _plot_dataset_line(ax_gap, ds, x, y):
                    _register_external_legend_dataset(ds, plotted_L_methods, plotted_methods)
                    plotted_gap.append(ds)
                    kappa_gap_values.append(x)
            if plotted_gap:
                theory_handles = []
                k_ref = _kappa_reference_grid(kappa_gap_values)
                if k_ref.size:
                    gap_ref = np.sqrt(1.0 + k_ref ** 2)
                    theory_line, = ax_gap.plot(
                        k_ref,
                        gap_ref,
                        color="0.35",
                        ls="--",
                        lw=1.6,
                        label=r"$\sqrt{1+\kappa^2}$",
                    )
                    theory_handles.append(theory_line)
                ax_gap.axvline(0.0, color="gray", ls=":", lw=1.1, alpha=0.75)
                _panel_setup(ax_gap, bc, xlabel=xlabel_kappa, ylabel=r"$\Delta_0(\kappa) / \Delta_0(0)$")
                ax_gap.set_title(f"{bc} (g={target_g})", loc="right", fontsize=TITLE_FONTSIZE)
                _theory_relation_legend(ax_gap, theory_handles, loc="upper center", fontsize=LEGEND_FONTSIZE - 2)
                autoscale_independent(ax_gap)
            else:
                _panel_setup(ax_gap, bc, xlabel=xlabel_kappa, ylabel=r"$\Delta_0(\kappa) / \Delta_0(0)$")
                ax_gap.set_title(f"{bc} (g={target_g})", loc="right", fontsize=TITLE_FONTSIZE)
                _no_data(ax_gap, "No valid data for collapse")

        else:
            for ax, xlab, ylab in (
                (ax_raw, xlabel_raw, r"$m_z$"),
                (ax_mz, xlabel_kappa, r"$m_z/m_0$"),
                (ax_gap, xlabel_kappa, r"$\Delta_0(\kappa) / \Delta_0(0)$"),
            ):
                _panel_setup(ax, bc, xlabel=xlab, ylabel=ylab)
                ax.set_title(f"{bc} (g={target_g})", loc="right", fontsize=TITLE_FONTSIZE)
                _no_data(ax, "No data")

        # Explicitly suppress x-ticks unless it's the last row
        if not is_last_row:
            for col in range(3):
                axes[row_idx, col].tick_params(labelbottom=False)

        # Ensure first-column x-range fixed as requested (-0.25, 0.25)
        axes[row_idx, 0].set_xlim((-0.25, 0.25))

        # Reduce y-axis label distance for the 1st and 2nd column
        axes[row_idx, 0].set_ylabel(r"$m_z$", fontsize=AXIS_LABEL_FONTSIZE, labelpad=-3)
        axes[row_idx, 1].set_ylabel(r"$m_z/m_0$", fontsize=AXIS_LABEL_FONTSIZE, labelpad=-3)
        for col in range(3):
            axes[row_idx, col].xaxis.label.set_size(AXIS_LABEL_FONTSIZE - 1)
            axes[row_idx, col].yaxis.label.set_size(AXIS_LABEL_FONTSIZE - 1)

    if not any_data:
        plt.close(fig)
        warn("FOQT scaling: no data available; combined figure skipped")
        record_figure("plots/hfield/foqt_scaling.pdf", False, "no data")
        return False


    # For a 4-row layout, an anchor around 0.11-0.12 works well for placing legend even closer to plots
    _external_size_method_legend(
        fig, plotted_L_methods, plotted_methods, y_anchor=0.145, fontsize=LEGEND_FONTSIZE - 4
    )

    save_fig(fig, "foqt_scaling.pdf")
    record_figure("plots/hfield/foqt_scaling.pdf", True)
    return True


# ---------------------------------------------------------------------------
# Gap report plot
# ---------------------------------------------------------------------------
def _gap_arrays_for_dataset(ds: HFieldDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = np.asarray(ds.data[:, HCOL["h"]], dtype=float)
    delta0 = np.asarray(ds.data[:, HCOL["delta_h"]], dtype=float)
    e0 = np.asarray(ds.data[:, HCOL["E0"]], dtype=float)
    e1 = np.asarray(ds.data[:, HCOL["E1"]], dtype=float)
    e2 = np.asarray(ds.data[:, HCOL["E2"]], dtype=float)

    delta0_from_energy = e1 - e0
    diff = np.abs(delta0 - delta0_from_energy)
    finite_diff = diff[np.isfinite(diff)]
    if finite_diff.size:
        max_diff = float(np.max(finite_diff))
        tol = 1e-10 if int(ds.method_code) == 0 else 1e-8
        if max_diff > tol:
            warn(
                f"{ds.path.name}: max |delta_h-(E1-E0)|={max_diff:.3e} "
                f"exceeds tolerance {tol:.1e}"
            )

    delta1 = e2 - e0
    return h, delta0, delta1


def _plot_single_gap_line(ax: plt.Axes, ds: HFieldDataset, gap_kind: str) -> bool:
    h, delta0, delta1 = _gap_arrays_for_dataset(ds)
    if gap_kind == "delta0":
        y = delta0
    elif gap_kind == "delta1":
        y = delta1
    else:
        raise ValueError(f"unknown gap_kind: {gap_kind}")

    m = np.isfinite(h) & np.isfinite(y) & (y > 0.0)
    if np.count_nonzero(m) < 1:
        return False

    h_use = h[m]
    y_use = y[m]
    order = np.argsort(h_use)
    h_use = h_use[order]
    y_use = y_use[order]

    colors = colors_for_bc(ds.bc_label)
    ax.plot(
        h_use,
        y_use,
        color=colors.get(ds.L, "gray"),
        ls="-",
        alpha=_line_alpha(ds),
        lw=1.35,
        label=_line_label(ds),
    )
    return True


def _gap_size_legend(ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    seen: set[str] = set()
    size_handles = []
    size_labels = []
    for handle, label in zip(handles, labels):
        if not label.startswith("$L=") or label in seen:
            continue
        seen.add(label)
        size_handles.append(handle)
        size_labels.append(label)
    if not size_handles:
        return
    ax.legend(
        size_handles,
        size_labels,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        ncol=2 if len(size_handles) > 4 else 1,
        loc="upper left",
    )


def plot_gaps_vs_h(
    cqt_data: dict[str, list[HFieldDataset]],
    foqt_data: dict[float, dict[str, list[HFieldDataset]]],
    foqt_g: list[float],
) -> bool:
    _ = foqt_g
    columns = [
        ("foqt", 0.5, r"$g=0.5$"),
        ("foqt", 0.9, r"$g=0.9$"),
        ("cqt", None, r"$g=g_{\rm pc}$"),
    ]
    rows = [
        ("PBC", "delta0", r"$\Delta_0$"),
        ("PBC", "delta1", r"$\Delta_1$"),
        ("OBC", "delta0", r"$\Delta_0$"),
        ("OBC", "delta1", r"$\Delta_1$"),
    ]

    fig, axes = plt.subplots(4, 3, figsize=(18, 14), constrained_layout=True)
    any_data = False
    plotted_L_methods: dict[int, set[int]] = {}
    plotted_methods: set[int] = set()

    for row_idx, (bc, gap_kind, ylabel) in enumerate(rows):
        for col_idx, (source, target_g, title) in enumerate(columns):
            ax = axes[row_idx, col_idx]
            if source == "cqt":
                datasets = list(cqt_data.get(bc, []))
                if not datasets:
                    warn(f"gaps_vs_h CQT {bc}: no datasets")
            else:
                assert target_g is not None
                datasets = list(foqt_data.get(float(target_g), {}).get(bc, []))
                if not datasets:
                    warn(f"gaps_vs_h FOQT {bc} g={_target_g_label(target_g)}: no datasets")

            plotted: list[HFieldDataset] = []
            for ds in datasets:
                if _plot_single_gap_line(ax, ds, gap_kind):
                    plotted.append(ds)
                    _register_external_legend_dataset(ds, plotted_L_methods, plotted_methods)

            ax.axvline(0.0, color="gray", ls=":", lw=1.1, alpha=0.75)
            ax.set_title(rf"{bc} ({title})", fontsize=TITLE_FONTSIZE)
            ax.set_xlabel(r"$h$", fontsize=AXIS_LABEL_FONTSIZE)
            ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
            apply_ticks(ax)
            apply_grid(ax)
            if plotted:
                any_data = True
                autoscale_independent(ax)
            else:
                _no_data(ax, "No data")
            if col_idx == 0:
                ax.set_xlim(-0.01, 0.01)
            if col_idx == 1:
                _col1_xlims = {0: (-0.01, 0.01), 1: (-0.02, 0.02), 2: (-0.01, 0.01), 3: (-0.02, 0.02)}
                ax.set_xlim(_col1_xlims[row_idx])
                _col1_ylims = {0: (0, 0.4), 1: (0.5, 3.2), 2: (0, 0.6), 3: (0.2, 2)}
                ax.set_ylim(_col1_ylims[row_idx])
            if col_idx == 2:
                ax.set(xlim=(-0.1, 0.1))
                _col2_ylims = {0: (0, 1.6), 1: (0.5, 3.5), 2: (0.1, 1.25), 3: (0.4, 2.3)}
                ax.set_ylim(_col2_ylims[row_idx])

    if not any_data:
        plt.close(fig)
        warn("gaps_vs_h: no finite positive gaps available; figure skipped")
        record_figure("plots/hfield/gaps_vs_h.pdf", False, "no data")
        return False

    _external_size_method_legend(
        fig,
        plotted_L_methods,
        plotted_methods,
        encode_method_in_size=False,
        y_anchor=-0.07,
    )
    save_fig(fig, "gaps_vs_h.pdf")
    record_figure("plots/hfield/gaps_vs_h.pdf", True)
    return True


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot h-field CQT/FOQT datasets produced by ising_hfield."
    )
    parser.add_argument("--L", nargs="+", type=int, default=DEFAULT_SIZES,
                        help="System sizes to include.")
    parser.add_argument("--only-cqt", action="store_true",
                        help="Deprecated (ignored): kept for compatibility.")
    parser.add_argument("--only-foqt", action="store_true",
                        help="Deprecated (ignored): kept for compatibility.")
    parser.add_argument("--foqt-g", nargs="+", type=float, default=[0.5, 0.9],
                        help="FOQT g values to include.")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Load partial files instead of skipping them.")
    parser.add_argument("--skip-susceptibility", action="store_true",
                        help="Skip Python chi_z computation and related CQT plots.")
    parser.add_argument("--no-clean-obsolete", action="store_true",
                        help="Do not remove obsolete PDF outputs.")
    parser.add_argument("--min-L-collapse", type=int, default=4,
                        help="Minimum L included in collapse plots.")
    parser.add_argument(
        "--use-local-extensions",
        action="store_true",
        help=(
            "Advanced local-only mode: read local extension data if present. "
            "Not needed for the public pipeline."
        ),
    )
    return parser


def apply_style() -> None:
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


def _syntax_check(path: Path) -> bool:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        critical(f"py_compile failed: cannot read {path}: {exc}")
        return False
    try:
        compile(source, str(path), "exec")
    except SyntaxError as exc:
        critical(f"py_compile failed: {exc}")
        return False
    return True


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _scan_hfield_pdfs() -> list[Path]:
    if not PLOT_DIR.exists():
        return []
    return sorted(PLOT_DIR.rglob("*.pdf"))


def _source_token_hits(tokens: tuple[str, ...]) -> list[tuple[int, str]]:
    try:
        lines = Path(__file__).resolve().read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        critical(f"source scan failed: {exc}")
        return []
    hits: list[tuple[int, str]] = []
    for lineno, line in enumerate(lines, start=1):
        if any(token in line for token in tokens):
            hits.append((lineno, line.strip()))
    return hits


def _print_final_report(*, py_compile_ok: bool, clean_skipped: bool,
                        discovered: list[Path], loaded: list[HFieldDataset],
                        cqt_data: dict[str, list[HFieldDataset]],
                        foqt_data: dict[float, dict[str, list[HFieldDataset]]],
                        skip_susceptibility: bool) -> None:
    all_pdfs = _scan_hfield_pdfs()
    final_present = [path for path in all_pdfs if path in FINAL_PDF_PATHS]
    diagnostic_pdfs = [path for path in all_pdfs if path not in FINAL_PDF_PATHS]
    missing_final = [path for path in sorted(FINAL_PDF_PATHS) if not path.exists()]
    shared_axis_hits = _source_token_hits(("share" + "x", "share" + "y"))
    manual_limit_hits = _source_token_hits(("set_" + "xlim", "set_" + "ylim"))

    print("\n[plot_hfield.py]  FINAL REPORT")
    print("1. FILE")
    print("  updated                 = scripts/h_field/plot_hfield.py")
    print(f"  py_compile              = {'PASS' if py_compile_ok else 'FAIL'}")

    print("2. DATASET")
    print(f"  files discovered        = {len(discovered)}")
    print(f"  datasets loaded         = {len(loaded)}")
    for bc in ("PBC", "OBC"):
        items = cqt_data.get(bc, [])
        n_partial = sum(1 for ds in items if ds.partial)
        print(f"  CQT {bc} datasets        = {len(items)} (partial={n_partial})")
    for target_g in sorted(foqt_data.keys()):
        for bc in ("PBC", "OBC"):
            items = foqt_data.get(float(target_g), {}).get(bc, [])
            n_partial = sum(1 for ds in items if ds.partial)
            print(
                f"  FOQT {bc} g={_target_g_label(target_g)} datasets = {len(items)} "
                f"(partial={n_partial})"
            )

    print("3. PDF FINALI PRESENTI")
    if final_present:
        for path in final_present:
            print(f"  - {_rel(path)}")
    else:
        print("  none")
    if missing_final:
        print("  missing:")
        for path in missing_final:
            print(f"  - {_rel(path)}")

    print("4. GAPS_VS_H")
    gap_path = PLOT_DIR / "gaps_vs_h.pdf"
    print(f"  gaps_vs_h.pdf          = {'present' if gap_path.exists() else 'missing'}")

    print("5. CONTROLLO ASSI CONDIVISI")
    if shared_axis_hits:
        print("  FAIL")
        for lineno, line in shared_axis_hits:
            print(f"  - line {lineno}: {line}")
    else:
        print("  PASS: nessun asse condiviso nei plot finali")

    print("6. CONTROLLO LIMITI MANUALI")
    if manual_limit_hits:
        print("  WARN")
        for lineno, line in manual_limit_hits:
            print(f"  - line {lineno}: {line}")
    else:
        print("  PASS: nessuna forzatura manuale dei limiti")

    print("7. PDF ELIMINATI")
    if clean_skipped:
        print("  skipped (--no-clean-obsolete)")
    elif OBSOLETE_REMOVED:
        for path in OBSOLETE_REMOVED:
            print(f"  - {_rel(path)}")
    else:
        print("  none")

    print("8. DIRECTORY OBSOLETE ELIMINATE")
    if clean_skipped:
        print("  skipped (--no-clean-obsolete)")
    elif REMOVED_DIRS:
        for path in REMOVED_DIRS:
            print(f"  - {_rel(path)}")
    else:
        print("  none")

    print("9. SUSCEPTIBILITY")
    print(f"  skipped                 = {skip_susceptibility}")
    print(f"  chi tables written      = {len(SAVED_TABLES)}")
    if SAVED_TABLES:
        for path in SAVED_TABLES:
            print(f"  - {_rel(path)}")

    print("10. WARNINGS")
    print(f"  count = {len(WARNINGS)}")
    for message in WARNINGS:
        print(f"  - {message}")
    if CRITICAL_ERRORS:
        print("  critical:")
        for message in CRITICAL_ERRORS:
            print(f"  - {message}")

    print("11. VERDICT")
    if diagnostic_pdfs:
        verdict = "FAIL"
    elif missing_final or shared_axis_hits:
        verdict = "FAIL"
    elif not py_compile_ok or CRITICAL_ERRORS:
        verdict = "FAIL"
    elif manual_limit_hits:
        verdict = "WARN"
    else:
        verdict = "PASS"
    print(f"  VERDETTO: {verdict}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.only_cqt or args.only_foqt:
        warn("--only-cqt/--only-foqt are deprecated and ignored")

    apply_style()
    py_compile_ok = _syntax_check(Path(__file__).resolve())

    print("[plot_hfield.py]  Configuration:")
    sizes = sorted(set(int(L) for L in args.L))
    foqt_g = sorted(set(float(g) for g in args.foqt_g))
    print(f"  sizes                 = {sizes}")
    print(f"  foqt_g                = {[_target_g_label(g) for g in foqt_g]}")
    print(f"  allow_partial         = {args.allow_partial}")
    print(f"  skip_susceptibility   = {args.skip_susceptibility}")
    print(f"  min_L_collapse        = {args.min_L_collapse}")
    print(f"  raw_dir               = {RAW_DIR.relative_to(PROJECT_ROOT)}")
    print(f"  plot_dir              = {PLOT_DIR.relative_to(PROJECT_ROOT)}")
    print(f"  clean_obsolete        = {not args.no_clean_obsolete}")
    if args.use_local_extensions:
        print("  local extension data enabled, if present")
    else:
        print("  local extension data disabled")

    if not RAW_DIR.exists():
        critical(f"raw directory does not exist: {RAW_DIR}")

    if args.no_clean_obsolete:
        print("[plot_hfield.py]  Obsolete PDF cleanup skipped")
        clean_skipped = True
    else:
        print("[plot_hfield.py]  Removing obsolete PDFs ...")
        clean_hfield_pdf_outputs()
        clean_skipped = False

    try:
        constants = load_fss_constants()
    except (OSError, KeyError, FileNotFoundError) as exc:
        critical(str(exc))
        _print_final_report(
            py_compile_ok=py_compile_ok,
            clean_skipped=clean_skipped,
            discovered=[],
            loaded=[],
            cqt_data={"PBC": [], "OBC": []},
            foqt_data={float(g): {"PBC": [], "OBC": []} for g in foqt_g},
            skip_susceptibility=bool(args.skip_susceptibility),
        )
        sys.exit(1)

    loaded = load_all_hfield_data(
        raw_dir=RAW_DIR,
        constants=constants,
        sizes=sizes,
        foqt_g=foqt_g,
        allow_partial=bool(args.allow_partial),
        do_cqt=True,
        do_foqt=True,
    )
    cqt_data = loaded["cqt"]
    foqt_data = loaded["foqt"]
    chi_by_bc: dict[str, list[dict]] = {"PBC": [], "OBC": []}

    print("\n[plot_hfield.py]  Generating report figures ...")
    plot_cqt_order_parameter(cqt_data, constants, min_L=int(args.min_L_collapse))

    if args.skip_susceptibility:
        warn("CQT susceptibility skipped by --skip-susceptibility")
        record_figure("plots/hfield/cqt_susceptibility.pdf", False, "skipped by option")
    else:
        if any(cqt_data.get(bc) for bc in ("PBC", "OBC")):
            print("\n[plot_hfield.py]  Computing CQT longitudinal susceptibility ...")
            chi_by_bc = build_chi_tables(cqt_data, constants)
            plot_cqt_susceptibility(chi_by_bc, constants, min_L=int(args.min_L_collapse))
        else:
            warn("CQT susceptibility: no CQT datasets available")
            record_figure("plots/hfield/cqt_susceptibility.pdf", False, "no data")

    if len(foqt_g) == 2:
        plot_combined_foqt_scaling(
            foqt_data,
            list(foqt_g),
            min_L=int(args.min_L_collapse),
            use_local_extensions=bool(args.use_local_extensions),
        )
    else:
        warn("foqt_g must have exactly 2 elements for combined plot")

    plot_gaps_vs_h(cqt_data, foqt_data, foqt_g)

    _print_final_report(
        py_compile_ok=py_compile_ok,
        clean_skipped=clean_skipped,
        discovered=loaded.get("discovered", []),
        loaded=loaded.get("all_loaded", []),
        cqt_data=cqt_data,
        foqt_data=foqt_data,
        skip_susceptibility=bool(args.skip_susceptibility),
    )


if __name__ == "__main__":
    main()
