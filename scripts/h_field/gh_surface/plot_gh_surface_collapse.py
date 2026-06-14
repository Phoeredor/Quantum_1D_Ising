#!/usr/bin/env python3
"""Plot multi-L collapse of scaled gh_surface datasets."""

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
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
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

GRID_TOL = 1.0e-10
ODDNESS_WARN = 1.0e-6
ODDNESS_FAIL = 1.0e-4

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 10,
    }
)


class CollapseError(RuntimeError):
    """Fatal validation error for collapse plotting."""


@dataclass
class RawDataset:
    path: Path
    meta: dict[str, str]
    columns: dict[str, int]
    data: np.ndarray


@dataclass
class SurfaceDataset:
    L: int
    path: Path
    rows: int
    kappa_g: np.ndarray
    kappa_h: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray
    max_oddness: float

    @property
    def grid_shape(self) -> tuple[int, int]:
        return self.Z.shape

    @property
    def z_min(self) -> float:
        return float(np.nanmin(self.Z))

    @property
    def z_max(self) -> float:
        return float(np.nanmax(self.Z))


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot multi-L collapse of scaled gh_surface datasets."
    )
    parser.add_argument("--L", nargs="+", type=int, default=[4, 6, 8, 10])
    parser.add_argument("--pbc", type=int, choices=(0, 1), default=1)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--plot-dir", type=Path, default=PLOT_DIR)
    parser.add_argument("--constants", type=Path, default=CONSTANTS_PATH)
    parser.add_argument("--reference-L", type=int, default=None)
    parser.add_argument("--downsample-surface", type=positive_int, default=4)
    parser.add_argument(
        "--sections-kh", nargs="+", type=float, default=[-4.0, 0.0, 4.0]
    )
    parser.add_argument(
        "--sections-kg", nargs="+", type=float, default=[-4.0, 0.0, 4.0]
    )
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def bc_name(pbc: int) -> str:
    return "PBC" if pbc else "OBC"


def bc_label(pbc: int) -> str:
    return "pbc" if pbc else "obc"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_constants(path: Path, bc: str) -> float:
    with path.open("r", encoding="utf-8") as fp:
        constants = json.load(fp)
    if bc not in constants:
        raise CollapseError(f"{rel(path)} is missing {bc} constants")
    if "beta_over_nu" not in constants[bc]:
        raise CollapseError(f"{rel(path)} {bc} entry is missing beta_over_nu")
    return float(constants[bc]["beta_over_nu"])


def parse_header(path: Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    header_lines = 0
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if line.startswith("#"):
                header_lines += 1
                text = line[1:].strip()
                if "=" in text:
                    key, value = text.split("=", 1)
                    meta[key.strip()] = value.strip()
                continue
            if line.strip():
                break
    if header_lines == 0:
        raise CollapseError(f"{rel(path)} has no header")
    return meta


def header_int(meta: dict[str, str], key: str) -> int | None:
    value = meta.get(key)
    if value is None:
        return None
    match = re.match(r"[-+]?\d+", value)
    if match is None:
        return None
    return int(match.group(0))


def columns_from_meta(path: Path, meta: dict[str, str]) -> dict[str, int]:
    raw = meta.get("columns")
    if raw is None:
        raise CollapseError(f"{rel(path)} header is missing columns")
    names = raw.split()
    if len(set(names)) != len(names):
        raise CollapseError(f"{rel(path)} header has duplicated column names")
    missing = [name for name in EXPECTED_COLUMNS if name not in names]
    if missing:
        raise CollapseError(
            f"{rel(path)} header is missing columns: {', '.join(missing)}"
        )
    return {name: names.index(name) for name in names}


def validate_metadata(ds: RawDataset, L: int, pbc: int) -> None:
    if ds.meta.get("grid_type") != "scaling":
        raise CollapseError(f"{rel(ds.path)} is not a scaling dataset")
    file_L = header_int(ds.meta, "L")
    file_pbc = header_int(ds.meta, "pbc")
    if file_L != L:
        raise CollapseError(f"{rel(ds.path)} has L={file_L}, expected L={L}")
    if file_pbc != pbc:
        raise CollapseError(f"{rel(ds.path)} has pbc={file_pbc}, expected pbc={pbc}")
    bc = ds.meta.get("BC")
    if bc is not None and bc.upper() != bc_name(pbc):
        raise CollapseError(f"{rel(ds.path)} has BC={bc}, expected {bc_name(pbc)}")


def load_raw_dataset(path: Path, L: int, pbc: int) -> RawDataset:
    meta = parse_header(path)
    columns = columns_from_meta(path, meta)
    try:
        data = np.loadtxt(path, comments="#")
    except ValueError as exc:
        raise CollapseError(f"{rel(path)} has malformed rows: {exc}") from exc
    if data.size == 0:
        raise CollapseError(f"{rel(path)} has no data rows")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != len(columns):
        raise CollapseError(
            f"{rel(path)} has {data.shape[1]} data columns, expected {len(columns)}"
        )
    ds = RawDataset(path=path, meta=meta, columns=columns, data=data)
    validate_metadata(ds, L, pbc)
    return ds


def nearest_index(values: np.ndarray, value: float, *, tol: float = GRID_TOL) -> int:
    hits = np.where(np.isclose(values, value, rtol=0.0, atol=tol))[0]
    if len(hits) != 1:
        raise CollapseError(f"value {value:.17e} does not map uniquely to grid")
    return int(hits[0])


def rectangular_grid(
    ds: RawDataset, z_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    kg = ds.data[:, ds.columns["kappa_g"]]
    kh = ds.data[:, ds.columns["kappa_h"]]
    kg_values = np.array(sorted(np.unique(kg)), dtype=float)
    kh_values = np.array(sorted(np.unique(kh)), dtype=float)
    if len(kg_values) * len(kh_values) != ds.data.shape[0]:
        raise CollapseError(
            f"{rel(ds.path)} is not rectangular: rows={ds.data.shape[0]}, "
            f"unique_grid={len(kh_values)}x{len(kg_values)}"
        )

    Z = np.full((len(kh_values), len(kg_values)), np.nan, dtype=float)
    filled = np.zeros_like(Z, dtype=bool)
    for kg_value, kh_value, z_value in zip(kg, kh, z_values):
        ix = nearest_index(kg_values, float(kg_value))
        iy = nearest_index(kh_values, float(kh_value))
        if filled[iy, ix]:
            raise CollapseError(
                f"{rel(ds.path)} has duplicated grid point "
                f"kappa_g={kg_value:.17e}, kappa_h={kh_value:.17e}"
            )
        Z[iy, ix] = float(z_value)
        filled[iy, ix] = True

    if not np.all(filled):
        missing = int(filled.size - np.count_nonzero(filled))
        raise CollapseError(f"{rel(ds.path)} has incomplete grid; missing={missing}")
    X, Y = np.meshgrid(kg_values, kh_values)
    return kg_values, kh_values, X, Y, Z


def max_oddness(kh_values: np.ndarray, Z: np.ndarray) -> float:
    values: list[float] = []
    missing = 0
    for iy, kh_value in enumerate(kh_values):
        # At exactly zero field the diagonalizer can return an arbitrary vector
        # in a degenerate parity sector, so test the nonzero +/- kh pairs.
        if np.isclose(kh_value, 0.0, rtol=0.0, atol=GRID_TOL):
            continue
        hits = np.where(np.isclose(kh_values, -kh_value, rtol=0.0, atol=GRID_TOL))[0]
        if len(hits) != 1:
            missing += 1
            continue
        values.append(float(np.nanmax(np.abs(Z[iy, :] + Z[int(hits[0]), :]))))
    if missing:
        raise CollapseError(f"cannot evaluate oddness: {missing} mirrored kh rows missing")
    return max(values) if values else float("nan")


def build_surface(ds: RawDataset, L: int, beta_over_nu: float) -> SurfaceDataset:
    mz = ds.data[:, ds.columns["mz"]]
    z_scaled = mz * (float(L) ** beta_over_nu)
    kg_values, kh_values, X, Y, Z = rectangular_grid(ds, z_scaled)
    oddness = max_oddness(kh_values, Z)
    return SurfaceDataset(
        L=L,
        path=ds.path,
        rows=ds.data.shape[0],
        kappa_g=kg_values,
        kappa_h=kh_values,
        X=X,
        Y=Y,
        Z=Z,
        max_oddness=oddness,
    )


def assert_common_grid(surfaces: list[SurfaceDataset]) -> tuple[np.ndarray, np.ndarray]:
    reference = surfaces[0]
    for surface in surfaces[1:]:
        same_shape = surface.grid_shape == reference.grid_shape
        same_kg = same_shape and np.allclose(
            surface.kappa_g, reference.kappa_g, rtol=0.0, atol=GRID_TOL
        )
        same_kh = same_shape and np.allclose(
            surface.kappa_h, reference.kappa_h, rtol=0.0, atol=GRID_TOL
        )
        if not (same_shape and same_kg and same_kh):
            raise CollapseError(
                "scaling grids do not coincide across L within tolerance; "
                "rigenera i dati con la stessa griglia in kappa_g,kappa_h"
            )
    return reference.kappa_g, reference.kappa_h


def downsample_grid(
    X: np.ndarray, Y: np.ndarray, Z: np.ndarray, step: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = list(range(0, X.shape[0], step))
    cols = list(range(0, X.shape[1], step))
    if rows[-1] != X.shape[0] - 1:
        rows.append(X.shape[0] - 1)
    if cols[-1] != X.shape[1] - 1:
        cols.append(X.shape[1] - 1)
    row_idx = np.array(rows, dtype=int)
    col_idx = np.array(cols, dtype=int)
    return X[np.ix_(row_idx, col_idx)], Y[np.ix_(row_idx, col_idx)], Z[np.ix_(row_idx, col_idx)]


def symmetric_levels(Z: np.ndarray, n_levels: int = 81) -> tuple[np.ndarray, TwoSlopeNorm]:
    vmax = float(np.nanmax(np.abs(Z)))
    if vmax == 0.0:
        vmax = 1.0
    levels = np.linspace(-vmax, vmax, n_levels)
    return levels, TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)


def section_indices(values: np.ndarray, requested: list[float]) -> list[tuple[float, int]]:
    sections: list[tuple[float, int]] = []
    for value in requested:
        hits = np.where(np.isclose(values, value, rtol=0.0, atol=GRID_TOL))[0]
        if len(hits) == 1:
            sections.append((float(value), int(hits[0])))
    if not sections:
        raise CollapseError(
            "none of the requested --sections-kh values are present in the common grid"
        )
    return sections


def make_output_path(plot_dir: Path, pbc: int, L_values: list[int]) -> Path:
    ltags = "_".join(f"L{L:02d}" for L in L_values)
    return plot_dir / f"mz_surface_scaling_collapse_{bc_label(pbc)}_{ltags}.pdf"


def plot_collapse(
    *,
    surfaces: list[SurfaceDataset],
    reference: SurfaceDataset,
    max_deviation: np.ndarray,
    pbc: int,
    out_path: Path,
    downsample_step: int,
    sections_kh: list[float],
    sections_kg: list[float],
) -> None:
    bc = bc_name(pbc)
    cmap_name = "plasma" if pbc else "viridis"
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0.18, 0.88, len(surfaces)))
    color_by_L = {surface.L: colors[i] for i, surface in enumerate(surfaces)}

    fig = plt.figure(figsize=(15.5, 11.0))
    gs = fig.add_gridspec(2, 2, wspace=0.25, hspace=0.28)

    ax0 = fig.add_subplot(gs[0, 0], projection="3d")
    for surface in surfaces:
        Xs, Ys, Zs = downsample_grid(surface.X, surface.Y, surface.Z, downsample_step)
        ax0.plot_surface(
            Xs,
            Ys,
            Zs,
            color=color_by_L[surface.L],
            linewidth=0.0,
            antialiased=True,
            alpha=0.40,
            shade=False,
        )
    ax0.set_xlabel(r"$\kappa_g$", labelpad=8)
    ax0.set_ylabel(r"$\kappa_h$", labelpad=8)
    ax0.set_zlabel(r"$m_z L^{\beta/\nu}$", labelpad=8)
    L_list = ",".join(str(surface.L) for surface in surfaces)
    ax0.set_title(f"{bc}: scaled surfaces, L={L_list}\nscaled-surface comparison")
    ax0.view_init(elev=25, azim=-135)
    ax0.tick_params(labelsize=10, pad=2)
    ax0.legend(
        handles=[
            Patch(facecolor=color_by_L[surface.L], edgecolor="none", label=f"L={surface.L}")
            for surface in surfaces
        ],
        loc="upper left",
        frameon=True,
        framealpha=0.85,
    )

    ax1 = fig.add_subplot(gs[0, 1])
    levels, norm = symmetric_levels(reference.Z)
    contour = ax1.contourf(
        reference.X,
        reference.Y,
        reference.Z,
        levels=levels,
        cmap=cmap_name,
        norm=norm,
        extend="both",
    )
    ax1.axvline(0.0, color="white", linewidth=1.1, linestyle="--", alpha=0.95)
    ax1.axhline(0.0, color="white", linewidth=1.1, linestyle="--", alpha=0.95)
    ax1.set_xlabel(r"$\kappa_g$")
    ax1.set_ylabel(r"$\kappa_h$")
    ax1.set_title(f"Reference scaled surface, $L_{{ref}}={reference.L}$")
    ax1.tick_params(direction="in", top=True, right=True)
    cb1 = fig.colorbar(contour, ax=ax1, fraction=0.046, pad=0.04)
    cb1.set_label(r"$m_z L^{\beta/\nu}$")

    ax2 = fig.add_subplot(gs[1, 0])
    max_residual = float(np.nanmax(max_deviation))
    residual_levels = np.linspace(0.0, max(max_residual, 1.0e-15), 81)
    contour_res = ax2.contourf(
        reference.X,
        reference.Y,
        max_deviation,
        levels=residual_levels,
        cmap="magma",
        extend="max",
    )
    ax2.axvline(0.0, color="white", linewidth=1.0, linestyle="--", alpha=0.85)
    ax2.axhline(0.0, color="white", linewidth=1.0, linestyle="--", alpha=0.85)
    ax2.set_xlabel(r"$\kappa_g$")
    ax2.set_ylabel(r"$\kappa_h$")
    ax2.set_title(r"Max deviation from $L_{ref}$")
    ax2.tick_params(direction="in", top=True, right=True)
    cb2 = fig.colorbar(contour_res, ax=ax2, fraction=0.046, pad=0.04)
    cb2.set_label(r"$\max_L |\Delta z_L|$")

    ax3 = fig.add_subplot(gs[1, 1])
    kh_sections = section_indices(reference.kappa_h, sections_kh)
    preferred_styles = ["-", "--", ":"]
    style_by_kh: dict[float, str] = {}
    for value, _ in kh_sections:
        if np.isclose(value, 0.0, rtol=0.0, atol=GRID_TOL):
            style_by_kh[value] = "-"
        else:
            used_nonzero = sum(
                not np.isclose(existing, 0.0, rtol=0.0, atol=GRID_TOL)
                for existing in style_by_kh
            )
            style_by_kh[value] = preferred_styles[(used_nonzero + 1) % len(preferred_styles)]
    for kg in sections_kg:
        if reference.kappa_g[0] <= kg <= reference.kappa_g[-1]:
            ax3.axvline(kg, color="0.86", linewidth=0.6, linestyle="-", zorder=0)
    for surface in surfaces:
        for kh_value, iy in kh_sections:
            ax3.plot(
                reference.kappa_g,
                surface.Z[iy, :],
                color=color_by_L[surface.L],
                linestyle=style_by_kh[kh_value],
                linewidth=1.6 if np.isclose(kh_value, 0.0, atol=GRID_TOL) else 1.15,
                alpha=0.96 if np.isclose(kh_value, 0.0, atol=GRID_TOL) else 0.70,
            )
    ax3.set_xlabel(r"$\kappa_g$")
    ax3.set_ylabel(r"$m_z L^{\beta/\nu}$")
    if len(kh_sections) == 1 and np.isclose(kh_sections[0][0], 0.0, atol=GRID_TOL):
        ax3.set_title(r"Section at $\kappa_h=0$")
    else:
        ax3.set_title(r"Sections at fixed $\kappa_h$")
    ax3.grid(True, color="0.88", linewidth=0.7)
    ax3.tick_params(direction="in", top=True, right=True)
    legend_L = ax3.legend(
        handles=[
            Line2D([0], [0], color=color_by_L[surface.L], lw=2.0, label=f"L={surface.L}")
            for surface in surfaces
        ],
        loc="upper left",
        frameon=True,
        framealpha=0.90,
        title="L",
    )
    ax3.add_artist(legend_L)
    if len(kh_sections) > 1:
        ax3.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color="0.2",
                    lw=1.8,
                    linestyle=style_by_kh[value],
                    label=rf"$\kappa_h={value:g}$",
                )
                for value, _ in kh_sections
            ],
            loc="lower right",
            frameon=True,
            framealpha=0.90,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_report(
    *,
    files_read: list[Path],
    surfaces: list[SurfaceDataset],
    pbc: int,
    beta_over_nu: float,
    reference: SurfaceDataset,
    kappa_g: np.ndarray,
    kappa_h: np.ndarray,
    max_dev_by_L: dict[int, float],
    out_path: Path,
    warnings: list[str],
    verdict: str,
) -> None:
    print("=== gh_surface collapse report ===")
    print("files_read:")
    for path in files_read:
        print(f"  - {rel(path)}")
    print("loaded_L = " + ",".join(str(surface.L) for surface in surfaces))
    print(f"BC = {bc_name(pbc)}")
    print(f"beta_over_nu = {beta_over_nu:.13g}")
    print(f"reference_L = {reference.L}")
    print(f"common_grid_shape = {len(kappa_h)} x {len(kappa_g)} (kappa_h x kappa_g)")
    print(f"kappa_g_range = [{kappa_g[0]:.8g}, {kappa_g[-1]:.8g}]")
    print(f"kappa_h_range = [{kappa_h[0]:.8g}, {kappa_h[-1]:.8g}]")
    print("per_L:")
    for surface in surfaces:
        ny, nx = surface.grid_shape
        print(
            f"  L={surface.L:02d} rows={surface.rows} grid_shape={ny}x{nx} "
            f"max_oddness={surface.max_oddness:.6e} "
            f"z_scaled_min={surface.z_min:.6e} z_scaled_max={surface.z_max:.6e}"
        )
    print("max_deviation_from_reference:")
    for L in sorted(max_dev_by_L):
        print(f"  L={L:02d} max_abs_delta_z={max_dev_by_L[L]:.6e}")
    if warnings:
        print("warnings:")
        for warning in warnings:
            print(f"  WARN {warning}")
    print(f"PDF = {rel(out_path)}")
    print(f"verdict = {verdict}")


def run(args: argparse.Namespace) -> str:
    bc = bc_name(args.pbc)
    beta_over_nu = load_constants(args.constants, bc)

    warnings: list[str] = []
    surfaces: list[SurfaceDataset] = []
    files_read: list[Path] = []
    requested_L = sorted(dict.fromkeys(args.L))

    for L in requested_L:
        path = args.raw_dir / f"ghsurf_scaling_{bc_label(args.pbc)}_L{L:02d}.dat"
        if not path.exists():
            message = f"missing requested file {rel(path)}"
            if args.allow_missing:
                print(f"WARN {message}; skipping L={L}")
                warnings.append(message)
                continue
            raise CollapseError(message)
        raw = load_raw_dataset(path, L, args.pbc)
        surface = build_surface(raw, L, beta_over_nu)
        surfaces.append(surface)
        files_read.append(path)

    if not surfaces:
        raise CollapseError("no scaling datasets loaded")

    kappa_g, kappa_h = assert_common_grid(surfaces)

    if args.reference_L is None:
        reference_L = max(surface.L for surface in surfaces)
    else:
        reference_L = args.reference_L
    reference_matches = [surface for surface in surfaces if surface.L == reference_L]
    if not reference_matches:
        raise CollapseError(f"reference-L={reference_L} was not loaded")
    reference = reference_matches[0]

    residuals = [
        np.abs(surface.Z - reference.Z) for surface in surfaces if surface.L != reference.L
    ]
    if residuals:
        max_deviation = np.maximum.reduce(residuals)
    else:
        max_deviation = np.zeros_like(reference.Z)
    max_dev_by_L = {
        surface.L: float(np.nanmax(np.abs(surface.Z - reference.Z)))
        for surface in surfaces
        if surface.L != reference.L
    }

    for surface in surfaces:
        if surface.max_oddness >= ODDNESS_FAIL:
            raise CollapseError(
                f"L={surface.L} oddness={surface.max_oddness:.6e} exceeds "
                f"FAIL threshold {ODDNESS_FAIL:.1e}"
            )
        if surface.max_oddness > ODDNESS_WARN:
            warnings.append(
                f"L={surface.L} oddness={surface.max_oddness:.6e} exceeds "
                f"{ODDNESS_WARN:.1e}"
            )

    z_scale = max(1.0, float(np.nanmax(np.abs(reference.Z))))
    large_deviation_threshold = 0.25 * z_scale
    for L, deviation in max_dev_by_L.items():
        if deviation > large_deviation_threshold:
            warnings.append(
                f"L={L} max deviation={deviation:.6e} is large relative to "
                f"L_ref={reference.L}"
            )

    out_path = make_output_path(args.plot_dir, args.pbc, [surface.L for surface in surfaces])
    plot_collapse(
        surfaces=surfaces,
        reference=reference,
        max_deviation=max_deviation,
        pbc=args.pbc,
        out_path=out_path,
        downsample_step=args.downsample_surface,
        sections_kh=args.sections_kh,
        sections_kg=args.sections_kg,
    )
    if not out_path.exists():
        raise CollapseError("no PDF produced")

    verdict = "WARN" if warnings else "PASS"
    print_report(
        files_read=files_read,
        surfaces=surfaces,
        pbc=args.pbc,
        beta_over_nu=beta_over_nu,
        reference=reference,
        kappa_g=kappa_g,
        kappa_h=kappa_h,
        max_dev_by_L=max_dev_by_L,
        out_path=out_path,
        warnings=warnings,
        verdict=verdict,
    )
    return verdict


def main() -> None:
    try:
        run(parse_args())
    except CollapseError as exc:
        print("=== gh_surface collapse report ===")
        print(f"FAIL {exc}")
        print("verdict = FAIL")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
