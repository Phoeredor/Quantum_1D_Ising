#!/usr/bin/env python3
"""Run, print, or emit h-field production grids from exported FSS constants."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import shlex
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONSTANTS_PATH = PROJECT_ROOT / "data" / "h_null" / "fss" / "fss_constants.json"
DEFAULT_L = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
STATIC_TOL = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch CQT/FOQT h-field production grids."
    )
    parser.add_argument("--constants", type=Path, default=CONSTANTS_PATH)
    parser.add_argument("--L", type=int, nargs="+", default=DEFAULT_L)
    parser.add_argument("--foqt-g", type=float, nargs="+", default=[0.5, 0.9])
    parser.add_argument("--xmax-cqt", type=float, default=12.0)
    parser.add_argument("--xmax-foqt", type=float, default=5.0)
    parser.add_argument("--dx-near", type=float, default=0.02)
    parser.add_argument("--dx-mid", type=float, default=0.10)
    parser.add_argument("--dx-far", type=float, default=0.50)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only-cqt", action="store_true")
    parser.add_argument("--only-foqt", action="store_true")
    parser.add_argument(
        "--emit-commands",
        type=Path,
        help="Write one independent command per output file and exit.",
    )
    parser.add_argument(
        "--single-L-commands",
        action="store_true",
        help="Generate one command per L instead of grouping L values.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-ram-gb", type=float)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of concurrent local jobs. Default: 1.",
    )
    parser.add_argument(
        "--nice-level",
        type=int,
        default=0,
        help="Nice level for local execution. Default: 0.",
    )
    return parser.parse_args()


def load_constants(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        constants = json.load(fp)
    for bc in ("PBC", "OBC"):
        if bc not in constants:
            raise KeyError(f"{path} does not contain a {bc} entry")
        for key in ("g_pc", "y_h"):
            if key not in constants[bc]:
                raise KeyError(f"{path} {bc} entry is missing {key}")
    return constants


def fmt_g(value: float) -> str:
    return format(value, ".17g")


def fmt_float(value: float) -> str:
    return format(value, ".17g")


def fmt_spacing(value: float) -> str:
    return fmt_float(value)


def adaptive_point_count(xmax: float, dx_near: float, dx_mid: float, dx_far: float) -> int:
    zones = [(0.0, 1.5, dx_near), (1.5, 5.0, dx_mid), (5.0, xmax, dx_far)]
    values: list[float] = []

    def add(value: float) -> None:
        if not any(abs(value - old) < 1e-10 for old in values):
            values.append(value)

    for x1, x2_default, dx in zones:
        x2 = x2_default
        if x1 > xmax:
            continue
        if x2 > xmax:
            x2 = xmax
        if x2 < x1:
            continue

        x = x1
        while x <= x2 + 0.5 * dx:
            x_use = min(x, x2)
            add(x_use)
            if abs(x_use) >= 1e-14:
                add(-x_use)
            if abs(x_use - x2) < 1e-14:
                break
            x += dx

    return len(sorted(values))


def static_gap_candidates(bc: str, L: int) -> list[Path]:
    base = PROJECT_ROOT / "data" / "static"
    if bc == "PBC":
        return [base / f"gap_L{L:02d}.dat", base / f"gap_lz_L{L:02d}.dat"]
    return [base / f"gap_obc_L{L:02d}.dat", base / f"gap_lz_obc_L{L:02d}.dat"]


def static_grid_has_g(path: Path, g_value: float, tol: float = STATIC_TOL) -> bool:
    try:
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                if not line.strip() or line.startswith("#"):
                    continue
                first = line.split(maxsplit=1)[0]
                try:
                    g = float(first)
                except ValueError:
                    continue
                if math.isfinite(g) and abs(g - g_value) <= tol:
                    return True
    except OSError:
        return False
    return False


def static_presence_for_bc(
    bc: str, g_value: float, L_values: list[int]
) -> tuple[bool, list[int]]:
    missing: list[int] = []
    for L in L_values:
        present = any(static_grid_has_g(path, g_value) for path in static_gap_candidates(bc, L))
        if not present:
            missing.append(L)
    return (len(missing) == 0), missing


def print_foqt_static_report(g_values: list[float], L_values: list[int]) -> None:
    print("FOQT static-grid presence report (tolerance 1e-8):")
    for g_value in g_values:
        pieces = []
        for bc in ("PBC", "OBC"):
            ok, missing = static_presence_for_bc(bc, g_value, L_values)
            status = "yes" if ok else "no"
            if missing:
                status += f" (missing L={','.join(str(L) for L in missing)})"
            pieces.append(f"{bc}: {status}")
        print(f"  g={fmt_g(g_value)}: " + "; ".join(pieces))


def command_to_string(cmd: list[str]) -> str:
    return shlex.join(cmd)


def run_command(cmd: list[str], nice_level: int) -> None:
    exec_cmd = cmd
    if nice_level != 0:
        exec_cmd = ["nice", "-n", str(nice_level), *cmd]
    subprocess.run(exec_cmd, cwd=PROJECT_ROOT, check=True)


def run_or_print(cmds: list[list[str]], dry_run: bool, jobs: int, nice_level: int) -> None:
    for cmd in cmds:
        print(command_to_string(cmd))

    if dry_run:
        return

    if jobs <= 1:
        for cmd in cmds:
            run_command(cmd, nice_level)
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(run_command, cmd, nice_level) for cmd in cmds]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def hfield_cmd(
    g_value: float,
    pbc: int,
    mode: str,
    L_values: list[int],
    xmax: float,
    dx_near: float,
    dx_mid: float,
    dx_far: float,
    yh: float | None = None,
    resume: bool = False,
    overwrite: bool = False,
    max_ram_gb: float | None = None,
) -> list[str]:
    cmd = [
        "./ising_hfield",
        fmt_g(g_value),
        str(pbc),
        "--mode",
        mode,
    ]
    if yh is not None:
        cmd.extend(["--yh", fmt_float(yh)])
    cmd.append("--L")
    cmd.extend(str(L) for L in L_values)
    cmd.extend(
        [
            "--xmax", fmt_float(xmax),
            "--dx-near", fmt_spacing(dx_near),
            "--dx-mid", fmt_spacing(dx_mid),
            "--dx-far", fmt_spacing(dx_far),
        ]
    )
    if resume:
        cmd.append("--resume")
    if overwrite:
        cmd.append("--overwrite")
    if max_ram_gb is not None:
        cmd.extend(["--max-ram-gb", fmt_float(max_ram_gb)])
    return cmd


def l_groups(L_values: list[int], split: bool) -> list[list[int]]:
    if split:
        return [[L] for L in L_values]
    return [L_values]


def build_commands(args: argparse.Namespace, constants: dict) -> list[tuple[str, list[str], int]]:
    do_cqt = not args.only_foqt
    do_foqt = not args.only_cqt
    split_l = args.single_L_commands or args.emit_commands is not None or args.jobs != 1
    commands: list[tuple[str, list[str], int]] = []

    if do_cqt:
        n_points = adaptive_point_count(
            float(args.xmax_cqt), float(args.dx_near), float(args.dx_mid), float(args.dx_far)
        )
        for bc, pbc in (("PBC", 1), ("OBC", 0)):
            g_pc = float(constants[bc]["g_pc"])
            y_h = float(constants[bc]["y_h"])
            for group in l_groups(args.L, split_l):
                cmd = hfield_cmd(
                    g_pc, pbc, "cqt", group,
                    xmax=float(args.xmax_cqt),
                    dx_near=float(args.dx_near),
                    dx_mid=float(args.dx_mid),
                    dx_far=float(args.dx_far),
                    yh=y_h,
                    resume=args.resume,
                    overwrite=args.overwrite,
                    max_ram_gb=args.max_ram_gb,
                )
                commands.append(("cqt", cmd, n_points * len(group)))

    if do_foqt:
        n_points = adaptive_point_count(
            float(args.xmax_foqt), float(args.dx_near), float(args.dx_mid), float(args.dx_far)
        )
        for g_value in args.foqt_g:
            for _bc, pbc in (("PBC", 1), ("OBC", 0)):
                for group in l_groups(args.L, split_l):
                    cmd = hfield_cmd(
                        float(g_value), pbc, "foqt", group,
                        xmax=float(args.xmax_foqt),
                        dx_near=float(args.dx_near),
                        dx_mid=float(args.dx_mid),
                        dx_far=float(args.dx_far),
                        resume=args.resume,
                        overwrite=args.overwrite,
                        max_ram_gb=args.max_ram_gb,
                    )
                    commands.append(("foqt", cmd, n_points * len(group)))

    return commands


def print_summary(commands: list[tuple[str, list[str], int]], args: argparse.Namespace) -> None:
    n_cqt = sum(1 for mode, _, _ in commands if mode == "cqt")
    n_foqt = sum(1 for mode, _, _ in commands if mode == "foqt")
    cqt_points = adaptive_point_count(
        float(args.xmax_cqt), float(args.dx_near), float(args.dx_mid), float(args.dx_far)
    )
    foqt_points = adaptive_point_count(
        float(args.xmax_foqt), float(args.dx_near), float(args.dx_mid), float(args.dx_far)
    )
    total_points = sum(points for _, _, points in commands)
    print("h-field command summary:")
    print(f"  commands        = {len(commands)}")
    print(f"  CQT commands    = {n_cqt}")
    print(f"  FOQT commands   = {n_foqt}")
    print("  L values        = " + " ".join(str(L) for L in args.L))
    print("  FOQT g values   = " + " ".join(fmt_g(float(g)) for g in args.foqt_g))
    print(f"  CQT points/cmd  = {cqt_points}")
    print(f"  FOQT points/cmd = {foqt_points}")
    print(f"  total points    = {total_points}")
    if args.emit_commands is None:
        print(f"  local jobs      = {args.jobs}")
        print(f"  local nice      = {args.nice_level}")
    else:
        print(f"  emit commands   = {args.emit_commands}")


def emit_commands(path: Path, commands: list[tuple[str, list[str], int]]) -> None:
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for _, cmd, _ in commands:
            fp.write(command_to_string(cmd) + "\n")
    print(f"Wrote {len(commands)} commands to {path}")


def main() -> int:
    args = parse_args()
    if args.only_cqt and args.only_foqt:
        raise SystemExit("[ERROR] --only-cqt and --only-foqt are mutually exclusive")
    if args.resume and args.overwrite:
        raise SystemExit("[ERROR] --resume and --overwrite are mutually exclusive")
    if args.xmax_cqt < 0.0 or args.xmax_foqt < 0.0:
        raise SystemExit("[ERROR] --xmax-cqt and --xmax-foqt must be nonnegative")
    if args.dx_near <= 0.0 or args.dx_mid <= 0.0 or args.dx_far <= 0.0:
        raise SystemExit("[ERROR] --dx-near, --dx-mid, and --dx-far must be positive")
    if args.max_ram_gb is not None and args.max_ram_gb <= 0.0:
        raise SystemExit("[ERROR] --max-ram-gb must be positive")
    if args.jobs <= 0:
        raise SystemExit("[ERROR] --jobs must be positive")

    constants_path = args.constants
    if not constants_path.is_absolute():
        constants_path = PROJECT_ROOT / constants_path
    constants = load_constants(constants_path)

    if not args.only_cqt:
        print_foqt_static_report(args.foqt_g, args.L)

    commands = build_commands(args, constants)
    print_summary(commands, args)

    if args.emit_commands is not None:
        emit_commands(args.emit_commands, commands)
        return 0

    run_or_print([cmd for _, cmd, _ in commands], args.dry_run, args.jobs, args.nice_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
