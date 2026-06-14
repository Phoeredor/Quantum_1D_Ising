#!/usr/bin/env python3
"""Check chi_z finite-difference g grids against their static sources."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path


OFFICIAL_L = (4, 6, 8, 10, 12, 14, 16, 18, 20, 22)
FILE_RE = re.compile(r"^chizfd(?P<obc>_obc)?_L(?P<L>\d{2})\.dat$")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def dh_tag(dh: float) -> str:
    sci = f"{dh:.12e}"
    mant, exp = sci.split("e", 1)
    mant = mant.rstrip("0").rstrip(".").replace(".", "p")
    return f"dh_{mant}e{exp}"


def method_code_for_l(L: int) -> int:
    return 0 if L <= 12 else 1


def source_candidates(root: Path, L: int, bc: str) -> list[Path]:
    pbc = bc == "PBC"
    method_code = method_code_for_l(L)
    names: list[str]

    if method_code == 0:
        if pbc:
            names = [f"gap_L{L:02d}.dat", f"gapL{L:02d}.dat"]
        else:
            names = [f"gap_obc_L{L:02d}.dat", f"gapobcL{L:02d}.dat"]
    else:
        if pbc:
            names = [f"gap_lz_L{L:02d}.dat", f"gaplzL{L:02d}.dat"]
        else:
            names = [f"gap_lz_obc_L{L:02d}.dat", f"gaplzobcL{L:02d}.dat"]

    bc_dir = "PBC" if pbc else "OBC"
    return [root / "data" / "h_null" / "observables" / bc_dir / name for name in names]


def chiz_bc_dir(out_dir: Path, bc: str) -> Path:
    return out_dir / ("PBC" if bc == "PBC" else "OBC")


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def parse_header(path: Path) -> dict[str, str]:
    header: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fp:
        for raw in fp:
            line = raw.strip()
            if not line.startswith("#"):
                break
            if "=" in line:
                key, value = line[1:].split("=", 1)
                header[key.strip()] = value.strip()
    return header


def load_first_col(path: Path) -> list[float]:
    vals: list[float] = []
    with path.open("r", encoding="utf-8") as fp:
        for line_no, raw in enumerate(fp, start=1):
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            token = stripped.split()[0]
            try:
                value = float(token)
            except ValueError as exc:
                raise ValueError(f"malformed first column in {path}:{line_no}") from exc
            if math.isfinite(value):
                vals.append(value)
    return vals


def discover_new_files(root: Path, out_dir: Path, require_all: bool) -> list[tuple[int, str, Path]]:
    entries: list[tuple[int, str, Path]] = []

    if require_all:
        for bc in ("PBC", "OBC"):
            for L in OFFICIAL_L:
                name = f"chizfd_L{L:02d}.dat" if bc == "PBC" else f"chizfd_obc_L{L:02d}.dat"
                entries.append((L, bc, chiz_bc_dir(out_dir, bc) / name))
        return entries

    for path in sorted(out_dir.glob("*/*chizfd*.dat")):
        match = FILE_RE.match(path.name)
        if not match:
            continue
        L = int(match.group("L"))
        bc = "OBC" if match.group("obc") else "PBC"
        entries.append((L, bc, path))

    return sorted(entries, key=lambda row: (row[1], row[0]))


def source_from_header_or_candidates(root: Path, new_file: Path, L: int, bc: str) -> Path | None:
    header = parse_header(new_file)
    raw_source = header.get("g_grid_source")
    if raw_source:
        source = Path(raw_source)
        if not source.is_absolute():
            source = root / source
        if source.exists():
            return source

    for candidate in source_candidates(root, L, bc):
        if candidate.exists():
            return candidate
    return None


def parse_float_pair(raw: str) -> tuple[float, float] | None:
    parts = raw.split()
    if len(parts) != 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def check_one(
    root: Path,
    L: int,
    bc: str,
    new_file: Path,
    allow_prefix: bool,
    allow_window: bool,
) -> bool:
    if not new_file.exists():
        print(
            f"L={L:02d} BC={bc:<3} new={rel(new_file, root)} "
            "source=<missing> n_new=0 n_source=0 max_abs_diff=nan FAIL"
        )
        return False

    header = parse_header(new_file)
    if (
        not allow_prefix
        and not allow_window
        and ("smoke_max_g_points" in header or "smoke_g_window" in header)
    ):
        print(
            f"L={L:02d} BC={bc:<3} new={rel(new_file, root)} "
            "source=<smoke-partial> n_new=0 n_source=0 "
            "max_abs_diff=nan mode=full SKIP"
        )
        return True

    source = source_from_header_or_candidates(root, new_file, L, bc)
    if source is None or not source.exists():
        source_label = "<missing>" if source is None else rel(source, root)
        n_new = len(load_first_col(new_file))
        print(
            f"L={L:02d} BC={bc:<3} new={rel(new_file, root)} "
            f"source={source_label} n_new={n_new} n_source=0 "
            "max_abs_diff=nan FAIL"
        )
        return False

    g_new = load_first_col(new_file)
    g_source = load_first_col(source)

    mode = "full"
    if allow_window and "smoke_g_window" in header:
        window = parse_float_pair(header["smoke_g_window"])
        if window is None:
            print(
                f"L={L:02d} BC={bc:<3} new={rel(new_file, root)} "
                f"source={rel(source, root)} n_new={len(g_new)} "
                f"n_source={len(g_source)} max_abs_diff=nan mode=window FAIL"
            )
            return False
        gmin, gmax = window
        g_source_cmp = [g for g in g_source if gmin <= g <= gmax]
        same_n = len(g_new) == len(g_source_cmp)
        mode = "window"
    elif allow_prefix:
        g_source_cmp = g_source
        same_n = len(g_new) <= len(g_source)
        mode = "prefix"
    else:
        g_source_cmp = g_source
        same_n = len(g_new) == len(g_source)

    if g_new and g_source_cmp:
        n_cmp = min(len(g_new), len(g_source_cmp))
        max_diff = max(abs(g_new[i] - g_source_cmp[i]) for i in range(n_cmp))
    else:
        max_diff = math.inf

    ok = same_n and max_diff < 1e-12
    status = "PASS" if ok else "FAIL"
    print(
        f"L={L:02d} BC={bc:<3} new={rel(new_file, root)} "
        f"source={rel(source, root)} n_new={len(g_new)} "
        f"n_source={len(g_source)} max_abs_diff={max_diff:.3e} "
        f"mode={mode} {status}"
    )
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare chi_z finite-difference g grids with static gap sources."
    )
    parser.add_argument("--dh", type=float, default=5e-4, help="Finite-difference dh value.")
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Require all official L=4..20 files for both PBC and OBC.",
    )
    parser.add_argument(
        "--allow-prefix",
        action="store_true",
        help="Allow new files to contain only a prefix of the static source grid.",
    )
    parser.add_argument(
        "--allow-window",
        action="store_true",
        help="Allow smoke g-window files and compare against the matching source window.",
    )
    args = parser.parse_args()

    root = repo_root()
    out_dir = root / "data" / "h_null" / "chiz_fd" / dh_tag(args.dh)
    entries = discover_new_files(root, out_dir, args.require_all)

    if not entries:
        print(f"No chizfd*.dat files found in {rel(out_dir, root)}")
        return 2

    all_ok = True
    for L, bc, new_file in entries:
        all_ok = check_one(root, L, bc, new_file, args.allow_prefix, args.allow_window) and all_ok

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
