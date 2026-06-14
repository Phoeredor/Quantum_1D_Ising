#!/usr/bin/env python3
"""
check_data_h_null.py
====================
Integrity checker for the h=0 static Ising data set.

The h=0 observables keep a reserved chi_z column for schema
compatibility, but the official longitudinal susceptibility is checked from
data/h_null/chiz_fd/<dh_tag>/<BC>/chizfd*.dat.

Usage:
    python3 scripts/h_null/check_data_h_null.py
    python3 scripts/h_null/check_data_h_null.py --static-dir data/h_null/observables
    python3 scripts/h_null/check_data_h_null.py --ed-sizes 4 6 8 10 12 --lz-sizes 14 16 18 20 22
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATIC_DIR = PROJECT_ROOT / "data" / "h_null" / "observables"
DEFAULT_CHIZ_ROOT = PROJECT_ROOT / "data" / "h_null" / "chiz_fd"
DEFAULT_DH = 5.0e-4

DEFAULT_ED_SIZES = [4, 6, 8, 10, 12]
DEFAULT_LZ_SIZES = [14, 16, 18, 20, 22]
DEFAULT_BCS = ["PBC", "OBC"]

STATUS_ORDER = [
    "MISSING",
    "EMPTY",
    "PARSE_ERROR",
    "WRONG_NCOLS",
    "OLD_SCHEMA",
    "TRUNCATED",
    "NONFINITE_MANDATORY",
    "NAN_CELLS",
    "INF_CELLS",
    "G_NOT_MONOTONE",
    "G_RANGE_WARN",
    "BAD_GAP_ORDER",
    "GAP_IDENTITY_FAIL",
    "MZSQ_NEG",
    "OBS_IDENTITY_FAIL",
    "BINDER_OOB",
    "PSIBAR_OOB",
    "HEADER_MISMATCH",
    "HEADER_NG_MISMATCH",
    "DH_MISMATCH",
    "METHOD_MISMATCH",
    "CHIFD_NONPOSITIVE",
    "ODDNESS_WARN",
    "GRID_MISSING_SOURCE",
    "GRID_MISSING_TARGET",
    "GRID_SIZE_MISMATCH",
    "GRID_MISMATCH",
]

HARD_ERROR_FLAGS = {
    "MISSING",
    "EMPTY",
    "PARSE_ERROR",
    "WRONG_NCOLS",
    "OLD_SCHEMA",
    "TRUNCATED",
    "NONFINITE_MANDATORY",
    "NAN_CELLS",
    "INF_CELLS",
    "G_NOT_MONOTONE",
    "BAD_GAP_ORDER",
    "GAP_IDENTITY_FAIL",
    "MZSQ_NEG",
    "OBS_IDENTITY_FAIL",
    "BINDER_OOB",
    "PSIBAR_OOB",
    "HEADER_MISMATCH",
    "HEADER_NG_MISMATCH",
    "DH_MISMATCH",
    "METHOD_MISMATCH",
    "CHIFD_NONPOSITIVE",
    "GRID_MISSING_SOURCE",
    "GRID_MISSING_TARGET",
    "GRID_SIZE_MISMATCH",
    "GRID_MISMATCH",
}

HEADER_RE = re.compile(r"^#\s*(?P<key>[A-Za-z0-9_ /.-]+?)\s*=\s*(?P<value>.*)$")


@dataclass
class FileRecord:
    group: str
    backend: str
    bc: str
    L: int
    ftype: str
    path: Path
    flags: list[str] = field(default_factory=list)
    exists: bool = False
    rows: int | None = None
    rows_valid: int | None = None
    ncols: int | None = None
    g_min: float | None = None
    g_max: float | None = None
    nan_count: int | None = None
    inf_count: int | None = None
    parse_error: str | None = None
    header: dict[str, str] = field(default_factory=dict, repr=False)
    arr: np.ndarray | None = field(default=None, repr=False)


@dataclass
class GridRecord:
    label: str
    backend: str
    bc: str
    L: int
    source: Path
    target: Path
    source_rows: int | None = None
    target_rows: int | None = None
    max_abs_diff: float | None = None
    flags: list[str] = field(default_factory=list)


def add_flag(flags: list[str], flag: str) -> None:
    if flag not in flags:
        flags.append(flag)


def ordered_flags(flags: list[str]) -> list[str]:
    return [flag for flag in STATUS_ORDER if flag in flags]


def status_text(flags: list[str]) -> str:
    ordered = ordered_flags(flags)
    return "OK" if not ordered else ",".join(ordered)


def has_errors(flags: list[str]) -> bool:
    return any(flag in HARD_ERROR_FLAGS for flag in flags)


def has_warnings(flags: list[str]) -> bool:
    return bool(flags) and not has_errors(flags)


def fmt_int(value: int | None) -> str:
    return "-" if value is None else str(value)


def fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    return f"{value:.{digits}f}"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def dh_tag(dh: float) -> str:
    sci = f"{dh:.12e}"
    mant, exp = sci.split("e", 1)
    mant = mant.rstrip("0").rstrip(".").replace(".", "p")
    return f"dh_{mant}e{exp}"


def method_code_for_l(L: int) -> int:
    return 0 if L <= 12 else 1


def static_filename(backend: str, bc: str, ftype: str, L: int) -> str:
    if backend == "ED":
        if ftype == "gap":
            return f"gap_L{L:02d}.dat" if bc == "PBC" else f"gap_obc_L{L:02d}.dat"
        return f"obs_L{L:02d}.dat" if bc == "PBC" else f"obs_obc_L{L:02d}.dat"

    if ftype == "gap":
        return f"gap_lz_L{L:02d}.dat" if bc == "PBC" else f"gap_lz_obc_L{L:02d}.dat"
    return f"obs_lz_L{L:02d}.dat" if bc == "PBC" else f"obs_lz_obc_L{L:02d}.dat"


def chiz_filename(bc: str, L: int) -> str:
    return f"chizfd_L{L:02d}.dat" if bc == "PBC" else f"chizfd_obc_L{L:02d}.dat"


def chiz_path(chiz_dir: Path, bc: str, L: int) -> Path:
    return chiz_dir / bc / chiz_filename(bc, L)


def static_path(static_dir: Path, backend: str, bc: str, ftype: str, L: int) -> Path:
    filename = static_filename(backend, bc, ftype, L)
    bc_dir = static_dir / bc
    if bc_dir.exists() or static_dir.name == "observables":
        return bc_dir / filename
    return static_dir / filename


def parse_header(path: Path) -> dict[str, str]:
    header: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if not line.startswith("#"):
                break
            match = HEADER_RE.match(line)
            if match:
                key = match.group("key").strip()
                if key not in header:
                    header[key] = match.group("value").strip()
    return header


def header_first_token_int(header: dict[str, str], key: str) -> int | None:
    raw = header.get(key)
    if raw is None:
        return None
    try:
        return int(raw.split()[0])
    except (ValueError, IndexError):
        return None


def header_float(header: dict[str, str], key: str) -> float | None:
    raw = header.get(key)
    if raw is None:
        return None
    try:
        return float(raw.split()[0])
    except (ValueError, IndexError):
        return None


def load_numeric(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, comments="#")
    arr = np.atleast_2d(arr)
    if arr.ndim != 2:
        raise ValueError(f"unexpected numeric array ndim={arr.ndim}")
    return arr


def base_file_record(group: str, backend: str, bc: str, L: int, ftype: str, path: Path) -> FileRecord:
    rec = FileRecord(group=group, backend=backend, bc=bc, L=L, ftype=ftype, path=path)
    if not path.exists():
        add_flag(rec.flags, "MISSING")
        return rec

    rec.exists = True
    if path.stat().st_size == 0:
        add_flag(rec.flags, "EMPTY")
        return rec

    try:
        rec.header = parse_header(path)
        rec.arr = load_numeric(path)
    except Exception as exc:  # pylint: disable=broad-except
        add_flag(rec.flags, "PARSE_ERROR")
        rec.parse_error = str(exc)
        return rec

    rec.rows = int(rec.arr.shape[0])
    rec.ncols = int(rec.arr.shape[1])
    rec.nan_count = int(np.isnan(rec.arr).sum())
    rec.inf_count = int(np.isinf(rec.arr).sum())
    if rec.nan_count > 0:
        add_flag(rec.flags, "NAN_CELLS")
    if rec.inf_count > 0:
        add_flag(rec.flags, "INF_CELLS")

    return rec


def fill_grid_summary(rec: FileRecord, mandatory_cols: list[int]) -> None:
    if rec.arr is None or rec.ncols is None:
        return
    if rec.ncols <= 0 or any(col >= rec.ncols for col in mandatory_cols):
        return

    mandatory = rec.arr[:, mandatory_cols]
    valid_rows = np.isfinite(mandatory).all(axis=1)
    rec.rows_valid = int(np.count_nonzero(valid_rows))
    if rec.rows_valid < 10:
        add_flag(rec.flags, "TRUNCATED")
    if rec.rows_valid != rec.rows:
        add_flag(rec.flags, "NONFINITE_MANDATORY")

    if rec.rows_valid == 0:
        return

    gvals = rec.arr[valid_rows, 0]
    rec.g_min = float(np.min(gvals))
    rec.g_max = float(np.max(gvals))
    if gvals.size >= 2 and not np.all(np.diff(gvals) > 0.0):
        add_flag(rec.flags, "G_NOT_MONOTONE")
    if rec.g_min > 0.55 or rec.g_max < 1.45:
        add_flag(rec.flags, "G_RANGE_WARN")


def check_gap_identities(rec: FileRecord, value_tol: float) -> None:
    if rec.arr is None or rec.ncols is None or rec.ncols not in (7, 8):
        return

    arr = rec.arr
    if rec.ncols == 8:
        e_cols = arr[:, 1:5]
        gap_col = 5
        gap_l_col = 6
        e0_l_col = 7
    else:
        e_cols = arr[:, 1:4]
        gap_col = 4
        gap_l_col = 5
        e0_l_col = 6

    if np.any(np.diff(e_cols, axis=1) < -value_tol):
        add_flag(rec.flags, "BAD_GAP_ORDER")

    gap = arr[:, gap_col]
    e0 = arr[:, 1]
    e1 = arr[:, 2]
    failures = [
        np.max(np.abs(gap - (e1 - e0))),
        np.max(np.abs(arr[:, gap_l_col] - rec.L * gap)),
        np.max(np.abs(arr[:, e0_l_col] - e0 / rec.L)),
    ]
    if any(np.isfinite(val) and val > value_tol for val in failures):
        add_flag(rec.flags, "GAP_IDENTITY_FAIL")


def check_obs_identities(rec: FileRecord, value_tol: float) -> None:
    if rec.arr is None or rec.ncols is None or rec.ncols < 11:
        return

    arr = rec.arr
    mz_sq = arr[:, 2]
    mz = arr[:, 3]
    psi_bar = arr[:, 7]
    binder = arr[:, 8]

    if np.any(mz_sq < -value_tol):
        add_flag(rec.flags, "MZSQ_NEG")

    mz_expected = np.sqrt(np.maximum(mz_sq, 0.0))
    if np.max(np.abs(mz - mz_expected)) > 5.0 * value_tol:
        add_flag(rec.flags, "OBS_IDENTITY_FAIL")

    if np.any((binder < 0.8) | (binder > 3.5)):
        add_flag(rec.flags, "BINDER_OOB")
    if np.any((psi_bar < -value_tol) | (psi_bar > 1.0 + value_tol)):
        add_flag(rec.flags, "PSIBAR_OOB")


def check_static_file(
    static_dir: Path,
    backend: str,
    bc: str,
    L: int,
    ftype: str,
    value_tol: float,
) -> FileRecord:
    path = static_path(static_dir, backend, bc, ftype, L)
    rec = base_file_record("static", backend, bc, L, ftype, path)
    if rec.arr is None or rec.ncols is None:
        return rec

    if ftype == "gap":
        expected_cols = 8 if backend == "ED" else 7
        if rec.ncols != expected_cols:
            add_flag(rec.flags, "WRONG_NCOLS")
        fill_grid_summary(rec, list(range(min(rec.ncols, expected_cols))))
        check_gap_identities(rec, value_tol)
        return rec

    if rec.ncols == 10:
        add_flag(rec.flags, "OLD_SCHEMA")
    elif rec.ncols != 11:
        add_flag(rec.flags, "WRONG_NCOLS")

    # chi_z is column 4 for schema compatibility; official data live in chiz_fd.
    mandatory_cols = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10] if rec.ncols >= 11 else list(range(rec.ncols))
    fill_grid_summary(rec, mandatory_cols)
    check_obs_identities(rec, value_tol)
    return rec


def check_chiz_file(
    chiz_dir: Path,
    static_source: FileRecord | None,
    backend: str,
    bc: str,
    L: int,
    dh: float,
    value_tol: float,
    oddness_tol: float,
) -> FileRecord:
    path = chiz_path(chiz_dir, bc, L)
    rec = base_file_record("chiz_fd", backend, bc, L, "chi_z", path)
    if rec.arr is None or rec.ncols is None:
        return rec

    if rec.ncols != 10:
        add_flag(rec.flags, "WRONG_NCOLS")
    fill_grid_summary(rec, list(range(min(rec.ncols, 10))))

    if rec.ncols >= 10:
        arr = rec.arr
        expected_method = method_code_for_l(L)
        methods = np.rint(arr[:, 2]).astype(int)
        if np.any(np.abs(arr[:, 2] - methods) > value_tol) or np.any(methods != expected_method):
            add_flag(rec.flags, "METHOD_MISMATCH")
        if not np.allclose(arr[:, 1], dh, rtol=0.0, atol=max(value_tol, 1e-15)):
            add_flag(rec.flags, "DH_MISMATCH")
        if np.any(arr[:, 7] <= 0.0):
            add_flag(rec.flags, "CHIFD_NONPOSITIVE")
        if np.max(np.abs(arr[:, 8:10])) > oddness_tol:
            add_flag(rec.flags, "ODDNESS_WARN")

    header_l = header_first_token_int(rec.header, "L")
    header_method = header_first_token_int(rec.header, "method_code")
    header_n_g = header_first_token_int(rec.header, "n_g")
    header_dh = header_float(rec.header, "dh")
    header_bc = rec.header.get("BC")

    if header_l is None or header_l != L:
        add_flag(rec.flags, "HEADER_MISMATCH")
    if header_bc is None or header_bc.split()[0] != bc:
        add_flag(rec.flags, "HEADER_MISMATCH")
    if header_method is None or header_method != method_code_for_l(L):
        add_flag(rec.flags, "HEADER_MISMATCH")
    if header_dh is None or abs(header_dh - dh) > max(value_tol, 1e-15):
        add_flag(rec.flags, "DH_MISMATCH")
    if header_n_g is None or (rec.rows is not None and header_n_g != rec.rows):
        add_flag(rec.flags, "HEADER_NG_MISMATCH")

    source_header = rec.header.get("g_grid_source")
    if source_header is None:
        add_flag(rec.flags, "HEADER_MISMATCH")
    elif static_source is not None:
        source_path = Path(source_header)
        if not source_path.is_absolute():
            source_path = PROJECT_ROOT / source_path
        if (
            source_path.resolve() != static_source.path.resolve()
            and source_path.name != static_source.path.name
        ):
            add_flag(rec.flags, "HEADER_MISMATCH")

    return rec


def compare_grid(
    label: str,
    source: FileRecord,
    target: FileRecord,
    g_tol: float,
    allow_prefix: bool = False,
) -> GridRecord:
    grid = GridRecord(
        label=label,
        backend=target.backend,
        bc=target.bc,
        L=target.L,
        source=source.path,
        target=target.path,
    )

    if source.arr is None:
        add_flag(grid.flags, "GRID_MISSING_SOURCE")
        return grid
    if target.arr is None:
        add_flag(grid.flags, "GRID_MISSING_TARGET")
        return grid

    g_source = source.arr[:, 0]
    g_target = target.arr[:, 0]
    grid.source_rows = int(g_source.size)
    grid.target_rows = int(g_target.size)

    if allow_prefix:
        same_size = g_target.size <= g_source.size
        g_source_cmp = g_source[: g_target.size]
    else:
        same_size = g_target.size == g_source.size
        g_source_cmp = g_source

    if not same_size:
        add_flag(grid.flags, "GRID_SIZE_MISMATCH")

    n_compare = min(g_target.size, g_source_cmp.size)
    if n_compare == 0:
        add_flag(grid.flags, "GRID_MISMATCH")
        return grid

    grid.max_abs_diff = float(np.max(np.abs(g_target[:n_compare] - g_source_cmp[:n_compare])))
    if grid.max_abs_diff > g_tol:
        add_flag(grid.flags, "GRID_MISMATCH")

    return grid


def build_static_records(
    static_dir: Path,
    ed_sizes: list[int],
    lz_sizes: list[int],
    bcs: list[str],
    value_tol: float,
) -> dict[tuple[str, str, int, str], FileRecord]:
    records: dict[tuple[str, str, int, str], FileRecord] = {}
    for backend, sizes in (("ED", ed_sizes), ("LZ", lz_sizes)):
        for bc in bcs:
            for L in sizes:
                for ftype in ("gap", "obs"):
                    rec = check_static_file(static_dir, backend, bc, L, ftype, value_tol)
                    records[(backend, bc, L, ftype)] = rec
    return records


def build_chiz_records(
    chiz_dir: Path,
    static_records: dict[tuple[str, str, int, str], FileRecord],
    ed_sizes: list[int],
    lz_sizes: list[int],
    bcs: list[str],
    dh: float,
    value_tol: float,
    oddness_tol: float,
) -> dict[tuple[str, str, int, str], FileRecord]:
    records: dict[tuple[str, str, int, str], FileRecord] = {}
    for backend, sizes in (("ED", ed_sizes), ("LZ", lz_sizes)):
        for bc in bcs:
            for L in sizes:
                source = static_records.get((backend, bc, L, "gap"))
                rec = check_chiz_file(chiz_dir, source, backend, bc, L, dh, value_tol, oddness_tol)
                records[(backend, bc, L, "chi_z")] = rec
    return records


def build_grid_records(
    static_records: dict[tuple[str, str, int, str], FileRecord],
    chiz_records: dict[tuple[str, str, int, str], FileRecord],
    ed_sizes: list[int],
    lz_sizes: list[int],
    bcs: list[str],
    g_tol: float,
    allow_partial_chizfd: bool,
) -> list[GridRecord]:
    grids: list[GridRecord] = []
    for backend, sizes in (("ED", ed_sizes), ("LZ", lz_sizes)):
        for bc in bcs:
            for L in sizes:
                gap = static_records[(backend, bc, L, "gap")]
                obs = static_records[(backend, bc, L, "obs")]
                chiz = chiz_records[(backend, bc, L, "chi_z")]
                grids.append(compare_grid("gap->obs", gap, obs, g_tol))
                grids.append(compare_grid("gap->chiz_fd", gap, chiz, g_tol, allow_partial_chizfd))
    return grids


def print_file_table(records: list[FileRecord]) -> None:
    print("FILE CHECKS")
    print(
        "group     backend  bc   L    type   status                         rows  valid  ncols   g_min   g_max  nans"
    )
    print(
        "--------  -------  ---  --   -----  -----------------------------  ----  -----  -----  ------  ------  ----"
    )
    for rec in records:
        print(
            f"{rec.group:<8}  {rec.backend:<7}  {rec.bc:<3}  {rec.L:>2d}   {rec.ftype:<5}  "
            f"{status_text(rec.flags):<29}  "
            f"{fmt_int(rec.rows):>4}  {fmt_int(rec.rows_valid):>5}  {fmt_int(rec.ncols):>5}  "
            f"{fmt_float(rec.g_min, 4):>6}  {fmt_float(rec.g_max, 4):>6}  "
            f"{fmt_int(rec.nan_count):>4}"
        )


def print_grid_table(records: list[GridRecord]) -> None:
    print("\nGRID CHECKS")
    print("label          backend  bc   L    status              n_src  n_tgt   max_abs_diff")
    print("-------------  -------  ---  --   ------------------  -----  -----  ------------")
    for rec in records:
        print(
            f"{rec.label:<13}  {rec.backend:<7}  {rec.bc:<3}  {rec.L:>2d}   "
            f"{status_text(rec.flags):<18}  {fmt_int(rec.source_rows):>5}  "
            f"{fmt_int(rec.target_rows):>5}  {fmt_float(rec.max_abs_diff, 3):>12}"
        )


def print_summary(files: list[FileRecord], grids: list[GridRecord], chiz_dir: Path) -> None:
    file_errors = [rec for rec in files if has_errors(rec.flags)]
    file_warnings = [rec for rec in files if has_warnings(rec.flags)]
    grid_errors = [rec for rec in grids if has_errors(rec.flags)]
    grid_warnings = [rec for rec in grids if has_warnings(rec.flags)]

    print("\nSUMMARY")
    print("-------")
    print(f"Observables/chiz_fd directory : {rel(chiz_dir)} / <BC>")
    print(f"Files expected           : {len(files)}")
    print(f"Files found              : {sum(1 for rec in files if rec.exists)}")
    print(f"Files OK                 : {sum(1 for rec in files if not rec.flags)}")
    print(f"Files with warnings      : {len(file_warnings)}")
    print(f"Files with errors        : {len(file_errors)}")
    print(f"Grid checks              : {len(grids)}")
    print(f"Grid warnings/errors     : {len(grid_warnings)} / {len(grid_errors)}")

    non_ok_files = [rec for rec in files if rec.flags]
    if non_ok_files:
        print("\nNon-OK files:")
        for rec in non_ok_files:
            msg = f"- {rel(rec.path)} :: {status_text(rec.flags)}"
            if rec.parse_error:
                msg += f" :: {rec.parse_error}"
            print(msg)

    non_ok_grids = [rec for rec in grids if rec.flags]
    if non_ok_grids:
        print("\nNon-OK grids:")
        for rec in non_ok_grids:
            print(
                f"- {rec.label} {rec.backend} {rec.bc} L={rec.L:02d}: "
                f"{status_text(rec.flags)} :: source={rel(rec.source)} target={rel(rec.target)}"
            )

    if not non_ok_files and not non_ok_grids:
        print("\nAll selected h=0 data are consistent and no sampled grid points are missing.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check h=0 static data, official finite-difference chi_z files, "
            "and point-by-point g-grid completeness."
        ),
    )
    parser.add_argument(
        "--static-dir",
        type=Path,
        default=DEFAULT_STATIC_DIR,
        help="Directory containing gap/obs h=0 data (default: PROJECT_ROOT/data/h_null/observables)",
    )
    parser.add_argument(
        "--chiz-root",
        type=Path,
        default=DEFAULT_CHIZ_ROOT,
        help="Root directory containing dh_* chi_z finite-difference folders.",
    )
    parser.add_argument(
        "--chiz-dh-tag",
        default=None,
        help="Subdirectory under --chiz-root. Defaults to the tag generated from --dh.",
    )
    parser.add_argument(
        "--dh",
        type=float,
        default=DEFAULT_DH,
        help="Finite-difference step expected in chiz_fd files (default: 5e-4).",
    )
    parser.add_argument(
        "--ed-sizes",
        nargs="+",
        type=int,
        default=DEFAULT_ED_SIZES,
        help="Full-ED sizes to check (default: 4 6 8 10 12).",
    )
    parser.add_argument(
        "--lz-sizes",
        nargs="+",
        type=int,
        default=DEFAULT_LZ_SIZES,
        help="Lanczos sizes to check (default: 14 16 18 20 22).",
    )
    parser.add_argument(
        "--bc",
        nargs="+",
        choices=DEFAULT_BCS,
        default=DEFAULT_BCS,
        help="Boundary conditions to check (default: PBC OBC).",
    )
    parser.add_argument(
        "--g-tol",
        type=float,
        default=1.0e-12,
        help="Absolute tolerance for g-grid equality (default: 1e-12).",
    )
    parser.add_argument(
        "--value-tol",
        type=float,
        default=5.0e-8,
        help="Absolute tolerance for algebraic consistency checks (default: 5e-8).",
    )
    parser.add_argument(
        "--oddness-tol",
        type=float,
        default=1.0e-8,
        help="Warning threshold for chiz_fd oddness columns (default: 1e-8).",
    )
    parser.add_argument(
        "--allow-partial-chizfd",
        action="store_true",
        help="Allow chiz_fd files to contain a prefix of the static g grid.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    static_dir = resolve_repo_path(args.static_dir)
    chiz_root = resolve_repo_path(args.chiz_root)
    chiz_dir = chiz_root / (args.chiz_dh_tag or dh_tag(args.dh))

    static_records = build_static_records(
        static_dir=static_dir,
        ed_sizes=args.ed_sizes,
        lz_sizes=args.lz_sizes,
        bcs=args.bc,
        value_tol=args.value_tol,
    )
    chiz_records = build_chiz_records(
        chiz_dir=chiz_dir,
        static_records=static_records,
        ed_sizes=args.ed_sizes,
        lz_sizes=args.lz_sizes,
        bcs=args.bc,
        dh=args.dh,
        value_tol=args.value_tol,
        oddness_tol=args.oddness_tol,
    )
    grids = build_grid_records(
        static_records=static_records,
        chiz_records=chiz_records,
        ed_sizes=args.ed_sizes,
        lz_sizes=args.lz_sizes,
        bcs=args.bc,
        g_tol=args.g_tol,
        allow_partial_chizfd=args.allow_partial_chizfd,
    )

    files = list(static_records.values()) + list(chiz_records.values())
    print_file_table(files)
    print_grid_table(grids)
    print_summary(files, grids, chiz_dir)

    hard_errors = any(has_errors(rec.flags) for rec in files) or any(has_errors(rec.flags) for rec in grids)
    return 1 if hard_errors else 0


if __name__ == "__main__":
    sys.exit(main())
