#!/usr/bin/env python3
"""Parallel L=22 chi_z finite-difference runner.

This script is intentionally separate from run_chifd_production.sh.  It uses
the exact g grids already present in data/h_null/observables/*/gap_lz*_L22.dat, runs one
independent ising_chiz_fd process per g point, and merges the validated
single-point outputs into the official chizfd_L22.dat files only at the end.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional


L = 22
METHOD_CODE = 1
DEFAULT_DH = "5e-4"
DEFAULT_JOBS = 8
DEFAULT_NICE = 19
EST_SEC_PER_POINT = {"PBC": 156.0, "OBC": 166.0}
CONTENTION_FACTOR = 1.6


@dataclass(frozen=True)
class GPoint:
    index: int
    text: str
    value: float


@dataclass(frozen=True)
class Task:
    bc: str
    point: GPoint
    task_dir: Path
    output_file: Path
    log_file: Path
    command: list[str]


@dataclass(frozen=True)
class TaskResult:
    task: Task
    elapsed: float
    skipped: bool


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def dh_tag(dh: str) -> str:
    mant, exp = f"{float(dh):.12e}".split("e", 1)
    mant = mant.rstrip("0").rstrip(".").replace(".", "p")
    return f"dh_{mant}e{exp}"


def bc_arg(bc: str) -> str:
    return "1" if bc == "PBC" else "0"


def final_name(bc: str) -> str:
    return "chizfd_L22.dat" if bc == "PBC" else "chizfd_obc_L22.dat"


def bc_output_dir(base: Path, bc: str) -> Path:
    return base / bc


def final_path(base: Path, bc: str) -> Path:
    return bc_output_dir(base, bc) / final_name(bc)


def source_gap_path(root: Path, bc: str) -> Path:
    name = "gap_lz_L22.dat" if bc == "PBC" else "gap_lz_obc_L22.dat"
    return root / "data" / "h_null" / "observables" / bc / name


def select_points(points: list[GPoint], max_points: int, probe_center: Optional[float]) -> list[GPoint]:
    if max_points <= 0 or max_points >= len(points):
        return points
    if probe_center is None:
        return points[:max_points]

    center_idx = min(range(len(points)), key=lambda i: abs(points[i].value - probe_center))
    start = center_idx - max_points // 2
    start = max(0, min(start, len(points) - max_points))
    return points[start : start + max_points]


def read_g_grid(path: Path) -> list[GPoint]:
    if not path.exists():
        raise FileNotFoundError(f"missing g-grid source: {path}")

    points: list[GPoint] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if not fields:
                continue
            text = fields[0]
            try:
                value = float(text)
            except ValueError as exc:
                raise ValueError(f"bad g value in {path}:{line_no}: {text!r}") from exc
            if not math.isfinite(value):
                raise ValueError(f"non-finite g value in {path}:{line_no}: {text!r}")
            points.append(GPoint(len(points), text, value))

    if not points:
        raise ValueError(f"no g points found in {path}")

    seen: set[str] = set()
    duplicates: list[str] = []
    for point in points:
        key = f"{point.value:.12g}"
        if key in seen:
            duplicates.append(point.text)
        seen.add(key)
    if duplicates:
        raise ValueError(f"duplicate g points in {path}: {', '.join(duplicates[:5])}")

    return points


def data_lines(path: Path) -> list[str]:
    rows: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                rows.append(stripped)
    return rows


def validate_row(row: str, point: GPoint, dh_value: float, source: Path) -> None:
    fields = row.split()
    if len(fields) != 10:
        raise ValueError(f"{source}: expected 10 columns, got {len(fields)}")

    try:
        g = float(fields[0])
        dh = float(fields[1])
        method = int(float(fields[2]))
        chi = float(fields[7])
        odd1 = float(fields[8])
        odd2 = float(fields[9])
    except ValueError as exc:
        raise ValueError(f"{source}: non-numeric row: {row}") from exc

    if not math.isfinite(g) or abs(g - point.value) > 5e-8:
        raise ValueError(f"{source}: g mismatch, expected {point.text}, got {fields[0]}")
    if not math.isfinite(dh) or abs(dh - dh_value) > max(1e-12, 1e-9 * dh_value):
        raise ValueError(f"{source}: dh mismatch, expected {dh_value}, got {fields[1]}")
    if method != METHOD_CODE:
        raise ValueError(f"{source}: method_code mismatch, expected {METHOD_CODE}, got {fields[2]}")
    if not math.isfinite(chi) or chi <= 0.0:
        raise ValueError(f"{source}: chi_fd must be finite and positive, got {fields[7]}")
    if not math.isfinite(odd1) or not math.isfinite(odd2):
        raise ValueError(f"{source}: non-finite oddness values")


def validate_point_output(path: Path, point: GPoint, dh_value: float) -> str:
    if not path.exists():
        raise FileNotFoundError(f"missing point output: {path}")
    rows = data_lines(path)
    if len(rows) != 1:
        raise ValueError(f"{path}: expected exactly one data row, got {len(rows)}")
    validate_row(rows[0], point, dh_value, path)
    return rows[0]


def validate_final_output(path: Path, points: list[GPoint], dh_value: float) -> bool:
    if not path.exists():
        return False
    try:
        rows = data_lines(path)
        if len(rows) != len(points):
            return False
        for row, point in zip(rows, points):
            validate_row(row, point, dh_value, path)
    except (OSError, ValueError):
        return False
    return True


def checkpoint_path(root: Path, args: argparse.Namespace, bc: str) -> Path:
    return root / args.work_dir / bc.lower() / "checkpoint.json"


def write_checkpoint(
    root: Path,
    args: argparse.Namespace,
    bc: str,
    source_grid: Path,
    points: list[GPoint],
    completed: set[int],
    failed: str | None = None,
) -> None:
    selected = select_points(points, args.max_points, args.probe_center)
    completed_sorted = sorted(completed)
    payload = {
        "schema": "chizfd_L22_parallel_checkpoint_v1",
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "bc": bc,
        "L": L,
        "dh": args.dh,
        "source_grid": str(source_grid.relative_to(root)),
        "work_dir": str((root / args.work_dir / bc.lower()).relative_to(root)),
        "output_file": str(final_path(root / args.output_dir, bc).relative_to(root)),
        "jobs": args.jobs,
        "nice": args.nice,
        "probe_mode": bool(args.max_points),
        "probe_center": args.probe_center,
        "n_source_points": len(points),
        "n_selected_points": len(selected),
        "n_completed": len(completed_sorted),
        "n_remaining": len(selected) - len(completed_sorted),
        "completed_indices": completed_sorted,
        "completed_g": [points[i].text for i in completed_sorted],
        "failed": failed,
        "resume_rule": "A point is considered complete only if its per-g output file validates.",
    }
    atomic_write_text(
        checkpoint_path(root, args, bc),
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def make_command(args: argparse.Namespace, root: Path, task_dir: Path, point: GPoint, bc: str) -> list[str]:
    binary = Path(args.binary)
    if not binary.is_absolute():
        binary = root / binary

    cmd: list[str] = []
    if not args.no_time:
        time_bin = Path("/usr/bin/time")
        if time_bin.exists():
            cmd.extend([str(time_bin), "-v"])
    if args.nice is not None:
        nice_bin = shutil.which("nice")
        if not nice_bin:
            raise FileNotFoundError("nice command not found")
        cmd.extend([nice_bin, "-n", str(args.nice)])
    cmd.extend(
        [
            str(binary),
            bc_arg(bc),
            args.dh,
            str(L),
            "--g-window",
            point.text,
            point.text,
            "--output-dir",
            str(task_dir),
            "--no-fsync",
        ]
    )
    return cmd


def build_tasks(args: argparse.Namespace, root: Path, bc: str, points: list[GPoint], log_dir: Path) -> list[Task]:
    work_dir = root / args.work_dir / bc.lower()
    selected = select_points(points, args.max_points, args.probe_center)
    tasks: list[Task] = []
    for point in selected:
        task_dir = work_dir / f"g_{point.index:03d}"
        output_file = final_path(task_dir, bc)
        log_file = log_dir / bc.lower() / f"g_{point.index:03d}.log"
        command = make_command(args, root, task_dir, point, bc)
        tasks.append(Task(bc, point, task_dir, output_file, log_file, command))
    return tasks


def command_preview(cmd: list[str], root: Path) -> str:
    parts = []
    for part in cmd:
        try:
            display = str(Path(part).resolve().relative_to(root))
        except (OSError, ValueError):
            display = part
        if any(ch.isspace() for ch in display):
            display = repr(display)
        parts.append(display)
    return " ".join(parts)


def run_task(task: Task, root: Path, dh_value: float, dry_run: bool) -> TaskResult:
    if not dry_run:
        try:
            validate_point_output(task.output_file, task.point, dh_value)
            return TaskResult(task=task, elapsed=0.0, skipped=True)
        except (OSError, ValueError):
            pass

    if dry_run:
        return TaskResult(task=task, elapsed=0.0, skipped=True)

    task.task_dir.mkdir(parents=True, exist_ok=True)
    task.log_file.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["BLIS_NUM_THREADS"] = "1"

    start = time.monotonic()
    with task.log_file.open("w", encoding="utf-8") as log:
        log.write("$ " + command_preview(task.command, root) + "\n\n")
        log.flush()
        proc = subprocess.run(
            task.command,
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.monotonic() - start
    if proc.returncode != 0:
        raise RuntimeError(
            f"{task.bc} g[{task.point.index}]={task.point.text} failed with exit={proc.returncode}; "
            f"see {task.log_file}"
        )
    validate_point_output(task.output_file, task.point, dh_value)
    return TaskResult(task=task, elapsed=elapsed, skipped=False)


def print_estimate(bc: str, n_remaining: int, jobs: int) -> tuple[float, float]:
    sec_per_point = EST_SEC_PER_POINT[bc]
    lower = n_remaining * sec_per_point / max(1, jobs)
    conservative = lower * CONTENTION_FACTOR
    print(
        f"  {bc}: remaining={n_remaining} jobs={jobs} "
        f"lower~{lower / 3600:.2f} h conservative~{conservative / 3600:.2f} h"
    )
    return lower, conservative


def write_merged_output(path: Path, bc: str, source_grid: Path, points: list[GPoint], rows: list[str], dh: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write("# chi_z finite-difference pipeline\n")
        fh.write(f"# dh = {float(dh):.12e}\n")
        fh.write(f"# L = {L}\n")
        fh.write(f"# BC = {bc}\n")
        fh.write(f"# method_code = {METHOD_CODE}\n")
        fh.write("# chi_fd = d <Mz/L> / dh at h=0\n")
        fh.write("# stencil = [-m(+2dh)+8m(+dh)-8m(-dh)+m(-2dh)]/(12dh)\n")
        fh.write("# method_code = 0 ED for L<=12, 1 Lanczos for L>=14\n")
        fh.write(f"# g_grid_source = {source_grid.as_posix()}\n")
        fh.write(f"# n_g = {len(points)}\n")
        fh.write("# L=22 generated by scripts/h_null/chiz_L22_parallel.py\n")
        fh.write("# NOT psi_tilde, NOT psi_bar, NOT sqrt(mz_sq), NOT observables COL[\"chi_z\"]\n")
        fh.write("# columns:\n")
        fh.write("# g dh method_code mz_m2 mz_m1 mz_p1 mz_p2 chi_fd oddness1 oddness2\n")
        for row in rows:
            fh.write(row.rstrip() + "\n")
    os.replace(tmp, path)


def merge_phase(args: argparse.Namespace, root: Path, bc: str, source_grid: Path, points: list[GPoint]) -> Path:
    dh_value = float(args.dh)
    rows: list[str] = []
    for point in points:
        output_file = final_path(root / args.work_dir / bc.lower() / f"g_{point.index:03d}", bc)
        rows.append(validate_point_output(output_file, point, dh_value))

    merged_path = final_path(root / args.output_dir, bc)
    write_merged_output(merged_path, bc, source_grid.relative_to(root), points, rows, args.dh)
    if not validate_final_output(merged_path, points, dh_value):
        raise RuntimeError(f"merged output failed validation: {merged_path}")
    return merged_path


def run_phase(args: argparse.Namespace, root: Path, bc: str, log_dir: Path) -> None:
    source_grid = source_gap_path(root, bc)
    points = read_g_grid(source_grid)
    dh_value = float(args.dh)
    final_file = final_path(root / args.output_dir, bc)

    if args.max_points:
        selected_points = select_points(points, args.max_points, args.probe_center)
        print(
            f"{bc}: smoke/probe mode, using {len(selected_points)}/{len(points)} g points; "
            "final merge disabled"
        )
        if args.probe_center is not None and selected_points:
            print(
                f"{bc}: probe window g=[{selected_points[0].text}, {selected_points[-1].text}] "
                f"around center={args.probe_center:g}"
            )
    elif validate_final_output(final_file, points, dh_value) and not args.force:
        print(f"{bc}: final output already complete, skipping: {final_file}")
        return

    tasks = build_tasks(args, root, bc, points, log_dir)
    n_already = 0
    completed_indices: set[int] = set()
    for task in tasks:
        try:
            validate_point_output(task.output_file, task.point, dh_value)
            n_already += 1
            completed_indices.add(task.point.index)
        except (OSError, ValueError):
            pass
    n_remaining = len(tasks) - n_already

    print(f"{bc}: grid={source_grid} points={len(points)} selected={len(tasks)} already={n_already}")
    print(f"{bc}: checkpoint={checkpoint_path(root, args, bc)}")
    if not args.dry_run:
        write_checkpoint(root, args, bc, source_grid, points, completed_indices)
    print_estimate(bc, n_remaining, min(args.jobs, max(1, len(tasks))))

    if args.dry_run:
        print(f"{bc}: dry-run commands (first 3):")
        for task in tasks[:3]:
            print("  " + command_preview(task.command, root))
        return

    completed = 0
    launched = time.monotonic()
    durations: list[float] = []
    max_workers = min(args.jobs, max(1, len(tasks)))
    with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(run_task, task, root, dh_value, False): task for task in tasks}
        for future in futures.as_completed(future_map):
            try:
                result = future.result()
            except Exception:
                for pending in future_map:
                    pending.cancel()
                failed_task = future_map[future]
                write_checkpoint(
                    root,
                    args,
                    bc,
                    source_grid,
                    points,
                    completed_indices,
                    failed=f"g[{failed_task.point.index}]={failed_task.point.text}",
                )
                raise
            completed += 1
            completed_indices.add(result.task.point.index)
            write_checkpoint(root, args, bc, source_grid, points, completed_indices)
            if not result.skipped:
                durations.append(result.elapsed)
            remaining = len(tasks) - completed
            avg = sum(durations) / len(durations) if durations else EST_SEC_PER_POINT[bc]
            eta = remaining * avg / max_workers
            phase_elapsed = time.monotonic() - launched
            status = "skip" if result.skipped else f"{result.elapsed / 60:.1f} min"
            print(
                f"{bc}: done {completed}/{len(tasks)} g[{result.task.point.index:03d}]="
                f"{result.task.point.text} {status}; elapsed={phase_elapsed / 3600:.2f} h "
                f"ETA~{eta / 3600:.2f} h"
            )
            sys.stdout.flush()

    if args.max_points:
        print(f"{bc}: probe outputs validated in {root / args.work_dir / bc.lower()}, final merge skipped")
        return

    final = merge_phase(args, root, bc, source_grid, points)
    print(f"{bc}: merged validated output -> {final}")


def parse_args() -> argparse.Namespace:
    root = repo_root_from_script()
    ap = argparse.ArgumentParser(
        description="Run L=22 chi_z FD point jobs in parallel, PBC then OBC by default."
    )
    ap.add_argument("--bc", choices=("PBC", "OBC", "both"), default="both")
    ap.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    ap.add_argument("--nice", type=int, default=DEFAULT_NICE)
    ap.add_argument("--dh", default=DEFAULT_DH)
    ap.add_argument("--root", type=Path, default=root)
    ap.add_argument("--binary", default="./ising_chiz_fd")
    ap.add_argument("--output-dir", type=Path, default=Path("data/h_null/chiz_fd") / dh_tag(DEFAULT_DH))
    ap.add_argument("--work-dir", type=Path, default=Path("data/h_null/chiz_fd/L22_parallel_tmp") / dh_tag(DEFAULT_DH))
    ap.add_argument("--log-dir", type=Path, default=None)
    ap.add_argument("--max-points", type=int, default=0, help="probe only: run first N points and skip final merge")
    ap.add_argument("--probe-center", type=float, default=None, help="with --max-points, choose a window around this g")
    ap.add_argument("--force", action="store_true", help="ignore an already complete final file and re-merge/re-run missing point jobs")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-time", action="store_true", help="do not wrap jobs with /usr/bin/time -v")
    args = ap.parse_args()

    if args.jobs <= 0:
        ap.error("--jobs must be positive")
    if args.nice is not None and not (0 <= args.nice <= 19):
        ap.error("--nice must be in [0,19]")
    try:
        dh_value = float(args.dh)
    except ValueError:
        ap.error("--dh must be numeric")
    if not math.isfinite(dh_value) or dh_value <= 0:
        ap.error("--dh must be finite and positive")
    if args.max_points < 0:
        ap.error("--max-points must be non-negative")
    if args.probe_center is not None and not math.isfinite(args.probe_center):
        ap.error("--probe-center must be finite")
    return args


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if not (root / "src" / "h_null" / "main_chiz_fd.c").exists():
        raise SystemExit(f"bad --root: {root}")
    binary = Path(args.binary)
    binary_path = binary if binary.is_absolute() else root / binary
    if not binary_path.exists() and not args.dry_run:
        raise SystemExit(f"missing binary {binary_path}; run: make chizfd")

    if args.log_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        log_dir = root / "logs" / "chiz_fd_L22_parallel" / stamp
    else:
        log_dir = args.log_dir if args.log_dir.is_absolute() else root / args.log_dir

    bcs = ["PBC", "OBC"] if args.bc == "both" else [args.bc]
    print("L=22 chi_z FD parallel runner")
    print(f"root      = {root}")
    print(f"binary    = {binary_path}")
    print(f"dh        = {args.dh}")
    print(f"jobs      = {args.jobs}")
    print(f"nice      = {args.nice}")
    print(f"log_dir   = {log_dir}")
    print(f"work_dir  = {root / args.work_dir}")
    print(f"output_dir= {root / args.output_dir}")
    print("phases    = " + " -> ".join(bcs))

    total_lower = 0.0
    total_cons = 0.0
    for bc in bcs:
        source = source_gap_path(root, bc)
        points = read_g_grid(source)
        final_file = final_path(root / args.output_dir, bc)
        if args.max_points:
            selected = min(args.max_points, len(points))
        elif validate_final_output(final_file, points, float(args.dh)) and not args.force:
            selected = 0
        else:
            selected = len(points)
        lower, cons = print_estimate(bc, selected, min(args.jobs, max(1, selected or 1)))
        total_lower += lower
        total_cons += cons
    print(f"estimated total, phases sequential: lower~{total_lower / 3600:.2f} h conservative~{total_cons / 3600:.2f} h")
    print()

    for bc in bcs:
        run_phase(args, root, bc, log_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
