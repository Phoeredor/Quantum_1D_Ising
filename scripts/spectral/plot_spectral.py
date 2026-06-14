#!/usr/bin/env python3
"""Plots and processed tables for generated spectral-statistics data."""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
G_REPORT = 1.0
BULK_FRAC = 0.5
OMEGA_DISTRIBUTIONS = 0.3
OMEGA_R_VS_H = 0.3
SPACING_KIND = "local"

H_GRID_R = (
    5e-4,
    1e-3,
    2e-3,
    5e-3,
    1e-2,
    2e-2,
    5e-2,
    1e-1,
    2e-1,
    5e-1,
)

H_DISTRIBUTIONS = (
    0.0,
    1e-3,
    5e-3,
    5e-2,
)

L_VALUES_R = (8, 10, 12)

MPLCONFIG_DIR = Path("/tmp/qising_1d_matplotlib_cache/spectral")
if "MPLCONFIGDIR" not in os.environ:
    MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(MPLCONFIG_DIR)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

RAW_DIR = PROJECT_ROOT / "data" / "spectral" / "raw"
RAW_BULK_DIR = RAW_DIR / "bulk0p5"
PROCESSED_DIR = PROJECT_ROOT / "data" / "spectral" / "processed"
PLOT_DIR = PROJECT_ROOT / "plots" / "spectral"

SPACING_H0_PDF = PLOT_DIR / "level_spacing_distribution_h0.pdf"
SPACING_NONZERO_PDF = PLOT_DIR / "level_spacing_distribution_h_nonzero.pdf"
RATIO_H0_PDF = PLOT_DIR / "ratio_distribution_h0.pdf"
RATIO_NONZERO_PDF = PLOT_DIR / "ratio_distribution_h_nonzero.pdf"
R_VS_H_PDF = PLOT_DIR / "spacing_ratio_vs_h_omega0p3.pdf"

R_TABLE = PROCESSED_DIR / "r_mean_vs_h_omega0p3.dat"
INVENTORY_TABLE = PROCESSED_DIR / "distribution_inventory.dat"
HISTOGRAM_TABLE = PROCESSED_DIR / "histogram_statistics.dat"

R_MEAN_POISSON = 2.0 * np.log(2.0) - 1.0
R_MEAN_GOE = 4.0 - 2.0 * np.sqrt(3.0)

POISSON_COLOR = "#0066ff"
GOE_COLOR = "#ff0000"
H0_HISTOGRAM_COLOR = "black"

AXIS_LABEL_FONTSIZE = 18
TICK_FONTSIZE = 15
LEGEND_FONTSIZE = 15
TITLE_FONTSIZE = 15
TEXT_FONTSIZE = 15

RAW_NAME_RE = re.compile(
    r"^spectral_pbc_L(?P<L>\d+)_g(?P<g>[^_]+)_omega(?P<omega>[^_]+)_"
    r"h(?P<h>[^_]+)_r(?P<realization>\d+)\.dat$"
)

S_HIST_RANGE = (0.0, 4.0)
R_HIST_RANGE = (0.0, 1.0)
DEFAULT_S_BINS = 48
DEFAULT_R_BINS = 40


@dataclass(frozen=True)
class HeaderMeta:
    L: int
    g: float
    omega: float
    h: float
    bulk_frac: float
    realization: int
    columns: tuple[str, ...]


@dataclass(frozen=True)
class DistributionSample:
    s_values: np.ndarray
    r_values: np.ndarray
    path: Path


@dataclass(frozen=True)
class ReadResult:
    path: Path
    status: str
    meta: HeaderMeta | None
    n_rows: int
    distribution_key: tuple[float, int, float] | None
    distribution_sample: DistributionSample | None
    r_key: tuple[int, float] | None
    r_mean: float | None
    n_r_values: int


@dataclass
class CollectedData:
    distribution: dict[tuple[float, int, float], dict[int, DistributionSample]]
    r_realizations: dict[tuple[int, float], dict[int, list[tuple[float, int, Path]]]]
    inventory_entries: list[ReadResult]
    warnings: list[str]
    candidate_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot generated spectral-statistics raw data."
    )
    parser.add_argument("--distribution-L", nargs="+", type=int, default=[12])
    parser.add_argument(
        "--spacing-kind",
        choices=("local", "global"),
        default=SPACING_KIND,
        help="Use s_local or s_global for P(s).",
    )
    parser.add_argument("--min-realizations-distribution", type=int, default=100)
    parser.add_argument("--min-realizations-r", type=int, default=1)
    parser.add_argument("--no-errorbars", action="store_true")
    args = parser.parse_args()

    if any(L not in L_VALUES_R for L in args.distribution_L):
        parser.error("--distribution-L may only contain 8, 10, 12")
    if args.min_realizations_distribution < 1:
        parser.error("--min-realizations-distribution must be >= 1")
    if args.min_realizations_r < 1:
        parser.error("--min-realizations-r must be >= 1")
    return args


def close_float(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def float_tag(value: float) -> str:
    return f"{value:.10g}"


def h_tolerance(target: float) -> float:
    return max(1e-12, 1e-10 * abs(target))


def match_target_h(value: float, targets: tuple[float, ...]) -> float | None:
    nearest = min(targets, key=lambda target: abs(value - target))
    if abs(value - nearest) <= h_tolerance(nearest):
        return nearest
    return None


def match_target_float(value: float, targets: tuple[float, ...]) -> float | None:
    for target in targets:
        if close_float(value, target):
            return target
    return None


def raw_name_metadata(path: Path) -> dict[str, float | int] | None:
    match = RAW_NAME_RE.match(path.name)
    if match is None:
        return None
    try:
        return {
            "L": int(match.group("L")),
            "g": float(match.group("g")),
            "omega": float(match.group("omega")),
            "h": float(match.group("h")),
            "realization": int(match.group("realization")),
        }
    except ValueError:
        return None


def path_matches_prefilter(
    path: Path,
    *,
    L_values: tuple[int, ...],
    omega_values: tuple[float, ...],
    h_values: tuple[float, ...],
) -> bool:
    metadata = raw_name_metadata(path)
    if metadata is None:
        return False
    if int(metadata["L"]) not in L_values:
        return False
    if not close_float(float(metadata["g"]), G_REPORT):
        return False
    if match_target_float(float(metadata["omega"]), omega_values) is None:
        return False
    return match_target_h(float(metadata["h"]), h_values) is not None


def iter_candidate_paths(
    *,
    distribution_omegas: tuple[float, ...],
    distribution_L_values: tuple[int, ...],
    include_r_grid: bool,
) -> list[Path]:
    L_values = tuple(sorted(set(distribution_L_values) | (set(L_VALUES_R) if include_r_grid else set())))
    omega_values = tuple(
        sorted(set(distribution_omegas) | ({OMEGA_R_VS_H} if include_r_grid else set()))
    )
    h_values = tuple(sorted(set(H_DISTRIBUTIONS) | (set(H_GRID_R) if include_r_grid else set())))

    paths: set[Path] = set()
    for raw_dir in (RAW_DIR, RAW_BULK_DIR):
        if not raw_dir.exists():
            continue
        with os.scandir(raw_dir) as entries:
            for entry in entries:
                if not entry.is_file() or not entry.name.endswith(".dat"):
                    continue
                path = Path(entry.path)
                if path_matches_prefilter(
                    path,
                    L_values=L_values,
                    omega_values=omega_values,
                    h_values=h_values,
                ):
                    paths.add(path.resolve())
    return sorted(paths)


def parse_header(path: Path) -> tuple[HeaderMeta | None, str]:
    header: dict[str, str] = {}
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                if not stripped.startswith("#"):
                    break
                payload = stripped[1:].strip()
                if "=" in payload:
                    key, value = payload.split("=", 1)
                    header[key.strip()] = value.strip()
    except OSError as exc:
        return None, f"read_failed:{exc}"

    required = {"L", "g", "omega", "h", "bulk_frac", "realization", "rng", "columns"}
    missing = sorted(required.difference(header))
    if missing:
        return None, "missing_header:" + ",".join(missing)
    if header["rng"] != "pcg32":
        return None, "unsupported_rng:" + header["rng"]

    try:
        meta = HeaderMeta(
            L=int(header["L"]),
            g=float(header["g"]),
            omega=float(header["omega"]),
            h=float(header["h"]),
            bulk_frac=float(header["bulk_frac"]),
            realization=int(header["realization"]),
            columns=tuple(header["columns"].split()),
        )
    except ValueError:
        return None, "invalid_numeric_header"
    return meta, "complete"


def column_index(meta: HeaderMeta, column: str) -> int:
    try:
        return meta.columns.index(column)
    except ValueError as exc:
        raise RuntimeError(f"missing column {column}") from exc


def finite_range(values: np.ndarray, low: float, high: float | None = None) -> np.ndarray:
    mask = np.isfinite(values) & (values >= low)
    if high is not None:
        mask &= values <= high
    return values[mask]


def load_values(
    path: Path,
    meta: HeaderMeta,
    *,
    spacing_kind: str,
    need_spacing: bool,
) -> tuple[np.ndarray | None, np.ndarray, int, str]:
    try:
        realization_col = column_index(meta, "realization")
        r_col = column_index(meta, "r")
        s_col = column_index(meta, f"s_{spacing_kind}") if need_spacing else None
    except RuntimeError as exc:
        return None, np.array([], dtype=float), 0, str(exc).replace(" ", "_")

    if need_spacing:
        usecols = (realization_col, int(s_col), r_col)
    else:
        usecols = (realization_col, r_col)

    try:
        arr = np.loadtxt(path, comments="#", dtype=float, ndmin=2, usecols=usecols)
    except ValueError:
        return None, np.array([], dtype=float), 0, "non_numeric_data"
    except OSError as exc:
        return None, np.array([], dtype=float), 0, f"read_failed:{exc}"

    if arr.size == 0:
        return None, np.array([], dtype=float), 0, "no_data_rows"
    n_rows = int(arr.shape[0])
    row_realizations = arr[:, 0].astype(int)
    if not np.all(row_realizations == meta.realization):
        return None, np.array([], dtype=float), n_rows, "header_row_realization_mismatch"

    if need_spacing:
        s_values = finite_range(arr[:, 1], 0.0, None)
        r_values = finite_range(arr[:, 2], R_HIST_RANGE[0], R_HIST_RANGE[1])
    else:
        s_values = None
        r_values = finite_range(arr[:, 1], R_HIST_RANGE[0], R_HIST_RANGE[1])
    return s_values, r_values, n_rows, "complete"


def classify_sample(
    meta: HeaderMeta,
    *,
    distribution_omegas: tuple[float, ...],
    distribution_L_values: tuple[int, ...],
    include_r_grid: bool,
) -> tuple[tuple[float, int, float] | None, tuple[int, float] | None]:
    if not close_float(meta.g, G_REPORT):
        return None, None
    if not close_float(meta.bulk_frac, BULK_FRAC):
        return None, None

    distribution_key = None
    matched_distribution_omega = match_target_float(meta.omega, distribution_omegas)
    matched_distribution_h = match_target_h(meta.h, H_DISTRIBUTIONS)
    if (
        matched_distribution_omega is not None
        and matched_distribution_h is not None
        and meta.L in distribution_L_values
    ):
        distribution_key = (matched_distribution_omega, meta.L, matched_distribution_h)

    r_key = None
    matched_r_h = match_target_h(meta.h, H_GRID_R)
    if (
        include_r_grid
        and meta.L in L_VALUES_R
        and close_float(meta.omega, OMEGA_R_VS_H)
        and matched_r_h is not None
    ):
        r_key = (meta.L, matched_r_h)

    return distribution_key, r_key


def read_candidate(
    path: Path,
    *,
    distribution_omegas: tuple[float, ...],
    distribution_L_values: tuple[int, ...],
    spacing_kind: str,
    include_r_grid: bool,
) -> ReadResult:
    meta, status = parse_header(path)
    if meta is None:
        return ReadResult(path, status, None, 0, None, None, None, None, 0)

    distribution_key, r_key = classify_sample(
        meta,
        distribution_omegas=distribution_omegas,
        distribution_L_values=distribution_L_values,
        include_r_grid=include_r_grid,
    )
    if distribution_key is None and r_key is None:
        return ReadResult(path, "header_outside_targets", meta, 0, None, None, None, None, 0)

    s_values, r_values, n_rows, data_status = load_values(
        path,
        meta,
        spacing_kind=spacing_kind,
        need_spacing=distribution_key is not None,
    )
    if data_status != "complete":
        return ReadResult(path, data_status, meta, n_rows, distribution_key, None, r_key, None, 0)

    distribution_sample = None
    if distribution_key is not None:
        assert s_values is not None
        distribution_sample = DistributionSample(s_values=s_values, r_values=r_values, path=path)

    r_mean = float(np.mean(r_values)) if r_key is not None and r_values.size else None
    return ReadResult(
        path,
        "complete",
        meta,
        n_rows,
        distribution_key,
        distribution_sample,
        r_key,
        r_mean,
        int(r_values.size) if r_key is not None else 0,
    )


def collect_data(
    *,
    distribution_omegas: tuple[float, ...],
    distribution_L_values: tuple[int, ...],
    spacing_kind: str,
    include_r_grid: bool,
) -> CollectedData:
    paths = iter_candidate_paths(
        distribution_omegas=distribution_omegas,
        distribution_L_values=distribution_L_values,
        include_r_grid=include_r_grid,
    )
    distribution: dict[tuple[float, int, float], dict[int, DistributionSample]] = {}
    r_realizations: dict[tuple[int, float], dict[int, list[tuple[float, int, Path]]]] = {}
    inventory_entries: list[ReadResult] = []
    warnings: list[str] = []
    duplicate_distribution_keys: set[tuple[float, int, float, int]] = set()

    read_workers = min(16, max(4, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=read_workers) as executor:
        results = executor.map(
            lambda candidate: read_candidate(
                candidate,
                distribution_omegas=distribution_omegas,
                distribution_L_values=distribution_L_values,
                spacing_kind=spacing_kind,
                include_r_grid=include_r_grid,
            ),
            paths,
        )
        for result in results:
            inventory_entries.append(result)
            if result.status != "complete" or result.meta is None:
                continue

            if result.distribution_key is not None and result.distribution_sample is not None:
                omega, L, h = result.distribution_key
                realization_key = (omega, L, h, result.meta.realization)
                group = distribution.setdefault(result.distribution_key, {})
                if result.meta.realization in group:
                    duplicate_distribution_keys.add(realization_key)
                else:
                    group[result.meta.realization] = result.distribution_sample

            if result.r_key is not None and result.r_mean is not None:
                per_realization = r_realizations.setdefault(result.r_key, {})
                per_realization.setdefault(result.meta.realization, []).append(
                    (result.r_mean, result.n_r_values, result.path)
                )

    for omega, L, h, realization in sorted(duplicate_distribution_keys)[:20]:
        warnings.append(
            f"duplicate distribution raw ignored after first sample: "
            f"omega={omega:.10g} L={L} h={h:.10g} realization={realization}"
        )
    if len(duplicate_distribution_keys) > 20:
        warnings.append(
            f"{len(duplicate_distribution_keys) - 20} additional duplicate distribution samples ignored"
        )

    return CollectedData(
        distribution=distribution,
        r_realizations=r_realizations,
        inventory_entries=inventory_entries,
        warnings=warnings,
        candidate_count=len(paths),
    )


def setup_style() -> None:
    plt.rcParams.update(
        {
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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def p_poisson_s(s: np.ndarray) -> np.ndarray:
    return np.exp(-s)


def p_goe_s(s: np.ndarray) -> np.ndarray:
    return 0.5 * np.pi * s * np.exp(-0.25 * np.pi * s * s)


def p_poisson_r(r: np.ndarray) -> np.ndarray:
    return 2.0 / (1.0 + r) ** 2


def p_goe_r(r: np.ndarray) -> np.ndarray:
    return (27.0 / 4.0) * (r + r * r) / (1.0 + r + r * r) ** 2.5


def draw_theory_s(ax: plt.Axes) -> None:
    grid = np.linspace(S_HIST_RANGE[0], S_HIST_RANGE[1], 500)
    ax.plot(grid, p_poisson_s(grid), color=POISSON_COLOR, ls="--", lw=2.0, label="Poisson")
    ax.plot(grid, p_goe_s(grid), color=GOE_COLOR, ls="--", lw=2.0, label="GOE")


def draw_theory_r(ax: plt.Axes) -> None:
    grid = np.linspace(R_HIST_RANGE[0], R_HIST_RANGE[1], 500)
    ax.plot(grid, p_poisson_r(grid), color=POISSON_COLOR, ls="--", lw=2.0, label="Poisson")
    ax.plot(grid, p_goe_r(grid), color=GOE_COLOR, ls="--", lw=2.0, label="GOE")


def h_legend_label(h: float) -> str:
    labels = {
        0.0: r"$h=0$",
        1e-3: r"$h=1\times 10^{-3}$",
        5e-3: r"$h= 5\times 10^{-3}$",
        5e-2: r"$h=5\times 10^{-2}$",
    }
    for value, label in labels.items():
        if close_float(h, value):
            return label
    return rf"$h={h:.0e}$"


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def raw_file_stem(L: int, g: float, omega: float, h: float, realization: int) -> str:
    return (
        f"spectral_pbc_L{L:02d}_g{float_tag(g)}_omega{float_tag(omega)}_"
        f"h{float_tag(h)}_r{realization:04d}"
    )


def expected_bulk_raw_path(L: int, omega: float, h: float, realization: int) -> Path:
    return RAW_BULK_DIR / f"{raw_file_stem(L, G_REPORT, omega, h, realization)}.dat"


def distribution_values(
    samples: dict[int, DistributionSample],
    observable: str,
) -> np.ndarray:
    if observable == "s":
        parts = [
            finite_range(sample.s_values, S_HIST_RANGE[0], S_HIST_RANGE[1])
            for sample in samples.values()
        ]
    elif observable == "r":
        parts = [finite_range(sample.r_values, R_HIST_RANGE[0], R_HIST_RANGE[1]) for sample in samples.values()]
    else:
        raise ValueError(f"unsupported observable {observable}")
    if not parts:
        return np.array([], dtype=float)
    return np.concatenate(parts) if len(parts) > 1 else parts[0]


def distribution_counts(samples: dict[int, DistributionSample]) -> tuple[int, int, int]:
    n_realizations = len(samples)
    n_s_values = int(
        sum(finite_range(sample.s_values, S_HIST_RANGE[0], S_HIST_RANGE[1]).size for sample in samples.values())
    )
    n_r_values = int(
        sum(finite_range(sample.r_values, R_HIST_RANGE[0], R_HIST_RANGE[1]).size for sample in samples.values())
    )
    return n_realizations, n_s_values, n_r_values


def status_for_count(n_realizations: int, min_realizations: int) -> str:
    if n_realizations == 0:
        return "missing"
    if n_realizations < min_realizations:
        return "below_min_realizations"
    return "used"


def write_distribution_inventory(
    collected: CollectedData,
    *,
    omega: float,
    distribution_L_values: tuple[int, ...],
    min_realizations: int,
) -> None:
    observed: set[tuple[int, float, int]] = set()

    def identity(result: ReadResult) -> tuple[int, float, int] | None:
        if result.meta is not None:
            L = result.meta.L
            h = match_target_h(result.meta.h, H_DISTRIBUTIONS)
            realization = result.meta.realization
            omega_match = close_float(result.meta.omega, omega)
        else:
            metadata = raw_name_metadata(result.path)
            if metadata is None:
                return None
            L = int(metadata["L"])
            h = match_target_h(float(metadata["h"]), H_DISTRIBUTIONS)
            realization = int(metadata["realization"])
            omega_match = close_float(float(metadata["omega"]), omega)
        if L not in distribution_L_values or h is None or not omega_match:
            return None
        return L, h, realization

    with INVENTORY_TABLE.open("w", encoding="utf-8") as f:
        f.write("# path status L omega h realization n_rows reason\n")
        for result in collected.inventory_entries:
            item = identity(result)
            if item is None:
                continue
            L, h, realization = item
            observed.add(item)
            if result.status == "complete" and result.distribution_key == (omega, L, h):
                status = "complete"
                reason = "-"
            elif result.status == "complete":
                status = "skipped"
                reason = "header_outside_distribution_targets"
            else:
                status = "invalid"
                reason = result.status
            f.write(
                f"{relative_path(result.path)} {status} {L} {omega:.17e} "
                f"{h:.17e} {realization} {result.n_rows} {reason}\n"
            )

        for L in distribution_L_values:
            for h in H_DISTRIBUTIONS:
                for realization in range(min_realizations):
                    item = (L, h, realization)
                    if item in observed:
                        continue
                    f.write(
                        f"{relative_path(expected_bulk_raw_path(L, omega, h, realization))} "
                        f"missing {L} {omega:.17e} {h:.17e} {realization} 0 "
                        "expected_by_min_realizations\n"
                    )


def write_histogram_statistics(rows: list[dict[str, object]]) -> None:
    with HISTOGRAM_TABLE.open("w", encoding="utf-8") as f:
        f.write("# observable L omega h n_realizations n_values bins mean std\n")
        for row in rows:
            mean = row["mean"]
            std = row["std"]
            mean_text = "nan" if mean is None else f"{float(mean):.17e}"
            std_text = "nan" if std is None else f"{float(std):.17e}"
            f.write(
                f"{row['observable']} {int(row['L'])} {float(row['omega']):.17e} "
                f"{float(row['h']):.17e} {int(row['n_realizations'])} "
                f"{int(row['n_hist_values'])} {int(row['n_bins'])} "
                f"{mean_text} {std_text}\n"
            )


def distribution_minimum_errors(
    collected: CollectedData,
    *,
    omega: float,
    distribution_L_values: tuple[int, ...],
    min_realizations: int,
) -> list[str]:
    errors: list[str] = []
    for L in distribution_L_values:
        for h in H_DISTRIBUTIONS:
            samples = collected.distribution.get((omega, L, h), {})
            n_realizations = len(samples)
            if n_realizations < min_realizations:
                errors.append(
                    f"L={L} omega={omega:.10g} h={h:.10g}: "
                    f"{n_realizations} realizations < "
                    f"min_realizations_distribution={min_realizations}"
                )
    return errors


def add_histogram_stat(
    stats: list[dict[str, object]],
    *,
    plot_name: str,
    observable: str,
    spacing_kind: str,
    omega: float,
    L: int,
    h: float,
    samples: dict[int, DistributionSample],
    values: np.ndarray,
    n_bins: int,
    min_realizations: int,
) -> str:
    n_real, n_s, n_r = distribution_counts(samples)
    status = status_for_count(n_real, min_realizations)
    mean = float(np.mean(values)) if values.size else None
    std = float(np.std(values, ddof=1)) if values.size > 1 else None
    stats.append(
        {
            "plot": plot_name,
            "observable": observable,
            "spacing_kind": spacing_kind,
            "omega": omega,
            "L": L,
            "h": h,
            "n_realizations": n_real,
            "n_s_values": n_s,
            "n_r_values": n_r,
            "n_hist_values": int(values.size),
            "n_bins": n_bins,
            "mean": mean,
            "std": std,
            "status": status,
        }
    )
    return status


def plot_distribution_pdf(
    collected: CollectedData,
    *,
    output_path: Path,
    plot_name: str,
    observable: str,
    omega: float,
    h_values: tuple[float, ...],
    distribution_L_values: tuple[int, ...],
    spacing_kind: str,
    min_realizations: int,
    histogram_stats: list[dict[str, object]],
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.6), constrained_layout=True)

    if observable == "s":
        draw_theory_s(ax)
        bins = np.linspace(S_HIST_RANGE[0], S_HIST_RANGE[1], DEFAULT_S_BINS + 1)
        ax.set_xlabel(r"$s$")
        ax.set_ylabel(r"$P(s)$")
        ax.set_xlim(*S_HIST_RANGE)
    elif observable == "r":
        draw_theory_r(ax)
        bins = np.linspace(R_HIST_RANGE[0], R_HIST_RANGE[1], DEFAULT_R_BINS + 1)
        ax.set_xlabel(r"$r$")
        ax.set_ylabel(r"$P(r)$")
        ax.set_xlim(*R_HIST_RANGE)
    else:
        raise ValueError(f"unsupported observable {observable}")

    if len(h_values) == 1 and close_float(h_values[0], 0.0):
        colors_by_h = {h_values[0]: H0_HISTOGRAM_COLOR}
    else:
        color_values = plt.cm.plasma(np.linspace(0.12, 0.86, max(1, len(h_values))))
        colors_by_h = {h: color_values[index] for index, h in enumerate(h_values)}

    legend_h_values: set[float] = set()
    plotted = 0

    for L in distribution_L_values:
        for h in h_values:
            samples = collected.distribution.get((omega, L, h), {})
            values = distribution_values(samples, observable)
            status = add_histogram_stat(
                histogram_stats,
                plot_name=plot_name,
                observable=observable,
                spacing_kind=spacing_kind,
                omega=omega,
                L=L,
                h=h,
                samples=samples,
                values=values,
                n_bins=len(bins) - 1,
                min_realizations=min_realizations,
            )
            if status != "used" or values.size == 0:
                continue

            label = h_legend_label(h) if h not in legend_h_values else "_nolegend_"
            legend_h_values.add(h)
            ax.hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                lw=2.0,
                color=colors_by_h[h],
                label=label,
            )
            plotted += 1

    if plotted == 0:
        ax.text(
            0.5,
            0.5,
            "no samples pass filters",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=TEXT_FONTSIZE,
        )

    h_text = "h=0" if len(h_values) == 1 and close_float(h_values[0], 0.0) else "h>0"
    ax.set_title(rf"$\omega={omega:.1f}$, {h_text}")
    ax.grid(alpha=0.25, ls=":")
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def summarize_r_grid(
    collected: CollectedData,
    *,
    min_realizations: int,
) -> tuple[list[dict[str, float]], list[str]]:
    rows: list[dict[str, float]] = []
    warnings: list[str] = []
    for L in L_VALUES_R:
        for h in H_GRID_R:
            per_realization = collected.r_realizations.get((L, h), {})
            realization_means: list[float] = []
            n_r_values = 0
            duplicate_count = 0
            for realization in sorted(per_realization):
                entries = per_realization[realization]
                if len(entries) > 1:
                    duplicate_count += len(entries) - 1
                realization_means.append(float(np.mean([entry[0] for entry in entries])))
                n_r_values += int(entries[0][1])

            n_realizations = len(realization_means)
            if duplicate_count:
                warnings.append(
                    f"duplicate r-grid raw averaged: L={L} h={h:.10g} duplicates={duplicate_count}"
                )
            if n_realizations < min_realizations:
                warnings.append(
                    f"skip r-grid row L={L} h={h:.10g}: "
                    f"{n_realizations} realizations < min_realizations={min_realizations}"
                )
                continue

            r_mean = float(np.mean(realization_means))
            r_sem = 0.0
            if n_realizations > 1:
                r_sem = float(np.std(realization_means, ddof=1) / np.sqrt(n_realizations))
            rows.append(
                {
                    "L": float(L),
                    "g": G_REPORT,
                    "omega": OMEGA_R_VS_H,
                    "bulk_frac": BULK_FRAC,
                    "h": h,
                    "n_realizations": float(n_realizations),
                    "r_mean": r_mean,
                    "r_sem_realization": r_sem,
                    "n_r_values": float(n_r_values),
                }
            )
    return rows, warnings


def write_r_table(rows: list[dict[str, float]]) -> None:
    with R_TABLE.open("w", encoding="utf-8") as f:
        f.write("# L g omega bulk_frac h n_realizations r_mean r_sem_realization n_r_values\n")
        for row in rows:
            f.write(
                f"{int(row['L'])} {row['g']:.17e} {row['omega']:.17e} "
                f"{row['bulk_frac']:.17e} {row['h']:.17e} "
                f"{int(row['n_realizations'])} {row['r_mean']:.17e} "
                f"{row['r_sem_realization']:.17e} {int(row['n_r_values'])}\n"
            )


def plot_r_vs_h(rows: list[dict[str, float]], *, show_errorbars: bool) -> None:
    del show_errorbars
    fig, ax = plt.subplots(figsize=(8.0, 5.6), constrained_layout=True)
    ax.axhline(
        R_MEAN_POISSON,
        color=POISSON_COLOR,
        ls="--",
        lw=2.0,
        label="_nolegend_",
    )
    ax.axhline(
        R_MEAN_GOE,
        color=GOE_COLOR,
        ls="--",
        lw=2.0,
        label="_nolegend_",
    )

    colors = plt.cm.viridis(np.linspace(0.12, 0.86, len(L_VALUES_R)))
    plotted = 0
    for color, L in zip(colors, L_VALUES_R):
        selected = [row for row in rows if int(row["L"]) == L and row["h"] > 0.0]
        selected.sort(key=lambda row: row["h"])
        if not selected:
            continue
        x = np.array([row["h"] for row in selected])
        y = np.array([row["r_mean"] for row in selected])
        ax.plot(x, y, marker="o", ms=5, lw=1.8, color=color, label=rf"$L={L}$")
        plotted += 1

    if plotted == 0:
        ax.text(
            0.5,
            0.5,
            "no <r>(h) rows pass filters",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=TEXT_FONTSIZE,
        )

    ax.set_xscale("log")
    ax.set_xlabel(r"$h$")
    ax.set_ylabel(r"$\langle r\rangle$")
    ax.set_ylim(0.30, 0.62)
    ax.text(
        0.03,
        R_MEAN_GOE + 0.012,
        "GOE",
        color=GOE_COLOR,
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=TEXT_FONTSIZE,
    )
    ax.text(
        0.03,
        R_MEAN_POISSON - 0.012,
        "Poisson",
        color=POISSON_COLOR,
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="top",
        fontsize=TEXT_FONTSIZE,
    )
    ax.grid(alpha=0.25, ls=":", which="both")
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE, ncol=1)
    fig.savefig(R_VS_H_PDF, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    omega_distribution = OMEGA_DISTRIBUTIONS
    distribution_L_values = tuple(args.distribution_L)

    collected = collect_data(
        distribution_omegas=(omega_distribution,),
        distribution_L_values=distribution_L_values,
        spacing_kind=args.spacing_kind,
        include_r_grid=True,
    )

    write_distribution_inventory(
        collected,
        omega=omega_distribution,
        distribution_L_values=distribution_L_values,
        min_realizations=args.min_realizations_distribution,
    )
    minimum_errors = distribution_minimum_errors(
        collected,
        omega=omega_distribution,
        distribution_L_values=distribution_L_values,
        min_realizations=args.min_realizations_distribution,
    )
    if minimum_errors:
        print("[spectral-report] ERROR: insufficient final distribution coverage")
        for error in minimum_errors:
            print("[spectral-report] ERROR:", error)
        print("[spectral-report] wrote:", INVENTORY_TABLE)
        return 1

    histogram_stats: list[dict[str, object]] = []
    plot_distribution_pdf(
        collected,
        output_path=SPACING_H0_PDF,
        plot_name="level_spacing_distribution_h0",
        observable="s",
        omega=omega_distribution,
        h_values=(0.0,),
        distribution_L_values=distribution_L_values,
        spacing_kind=args.spacing_kind,
        min_realizations=args.min_realizations_distribution,
        histogram_stats=histogram_stats,
    )
    plot_distribution_pdf(
        collected,
        output_path=SPACING_NONZERO_PDF,
        plot_name="level_spacing_distribution_h_nonzero",
        observable="s",
        omega=omega_distribution,
        h_values=(1e-3, 5e-3, 5e-2),
        distribution_L_values=distribution_L_values,
        spacing_kind=args.spacing_kind,
        min_realizations=args.min_realizations_distribution,
        histogram_stats=histogram_stats,
    )
    plot_distribution_pdf(
        collected,
        output_path=RATIO_H0_PDF,
        plot_name="ratio_distribution_h0",
        observable="r",
        omega=omega_distribution,
        h_values=(0.0,),
        distribution_L_values=distribution_L_values,
        spacing_kind=args.spacing_kind,
        min_realizations=args.min_realizations_distribution,
        histogram_stats=histogram_stats,
    )
    plot_distribution_pdf(
        collected,
        output_path=RATIO_NONZERO_PDF,
        plot_name="ratio_distribution_h_nonzero",
        observable="r",
        omega=omega_distribution,
        h_values=(1e-3, 5e-3, 5e-2),
        distribution_L_values=distribution_L_values,
        spacing_kind=args.spacing_kind,
        min_realizations=args.min_realizations_distribution,
        histogram_stats=histogram_stats,
    )

    write_histogram_statistics(histogram_stats)

    r_rows, r_warnings = summarize_r_grid(
        collected,
        min_realizations=args.min_realizations_r,
    )
    write_r_table(r_rows)
    plot_r_vs_h(r_rows, show_errorbars=False)

    print("[spectral-report] candidate raw files:", collected.candidate_count)
    print(f"[spectral-report] distribution omega: {omega_distribution:.10g}")
    print(f"[spectral-report] spacing kind: {args.spacing_kind}")
    print("[spectral-report] wrote:", SPACING_H0_PDF)
    print("[spectral-report] wrote:", SPACING_NONZERO_PDF)
    print("[spectral-report] wrote:", RATIO_H0_PDF)
    print("[spectral-report] wrote:", RATIO_NONZERO_PDF)
    print("[spectral-report] wrote:", R_VS_H_PDF)
    print("[spectral-report] wrote:", R_TABLE)
    print("[spectral-report] wrote:", INVENTORY_TABLE)
    print("[spectral-report] wrote:", HISTOGRAM_TABLE)
    print("[spectral-report] r errorbars: disabled")
    for row in histogram_stats:
        print(
            "[spectral-report] histogram "
            f"plot={row['plot']} observable={row['observable']} "
            f"L={row['L']} h={float(row['h']):.10g} "
            f"N={row['n_realizations']} n={row['n_hist_values']} "
            f"status={row['status']}"
        )
    for warning in collected.warnings + r_warnings:
        print("[spectral-report] WARNING:", warning)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
