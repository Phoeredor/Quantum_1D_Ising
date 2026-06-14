#!/usr/bin/env python3
"""Build, shard, run, benchmark, and merge two-parameter (g,h) surfaces."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import shutil
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONSTANTS_PATH = PROJECT_ROOT / "data" / "h_null" / "fss" / "fss_constants.json"
BASE_DIR = PROJECT_ROOT / "data" / "h_field" / "gh_surface"
GRID_DIR = BASE_DIR / "grids"
RAW_DIR = BASE_DIR / "raw"
WORK_DIR = BASE_DIR / "work"
PROCESSED_DIR = BASE_DIR / "processed"
PLOT_DIR = PROJECT_ROOT / "plots" / "hfield" / "gh_surface"
EXECUTABLE = PROJECT_ROOT / "ising_gh_surface"

THREAD_ENV_VARS = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]
DATA_COLUMNS = [
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
PRODUCTION_POINT_THRESHOLD = 1000
GRID_TOL = 1e-10


@dataclass(frozen=True)
class GridSpec:
    grid_type: str
    grid_path: Path
    out_path: Path
    rows: list[tuple[float, float, float, float]]

    @property
    def n_points(self) -> int:
        return len(self.rows)


@dataclass(frozen=True)
class ShardSpec:
    grid_type: str
    index: int
    grid_path: Path
    out_path: Path
    log_path: Path
    rows: list[tuple[float, float, float, float]]

    @property
    def n_points(self) -> int:
        return len(self.rows)


@dataclass(frozen=True)
class ResourceInfo:
    hostname: str
    cpu_model: str
    logical_cores: int
    load_avg: tuple[float, float, float] | None
    mem_total_gib: float | None
    mem_available_gib: float | None
    swap_total_gib: float | None
    swap_used_gib: float | None
    disk_available_gib: float
    thread_env: dict[str, str | None]
    recommended_jobs: int
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate physical and CQT scaling (g,h) surface data."
    )
    parser.add_argument("--L", type=int, default=8)
    parser.add_argument("--pbc", type=int, choices=(0, 1), default=1)
    parser.add_argument("--grid-type", choices=("physical", "scaling", "both"), default="both")
    parser.add_argument("--g-min", type=float, default=0.4)
    parser.add_argument("--g-max", type=float, default=1.6)
    parser.add_argument("--g-points", type=int, default=81)
    parser.add_argument("--h-min", type=float, default=-0.20)
    parser.add_argument("--h-max", type=float, default=0.20)
    parser.add_argument("--h-points", type=int, default=81)
    parser.add_argument("--kg-min", type=float, default=-8.0)
    parser.add_argument("--kg-max", type=float, default=8.0)
    parser.add_argument("--kg-points", type=int, default=65)
    parser.add_argument("--kh-min", type=float, default=-8.0)
    parser.add_argument("--kh-max", type=float, default=8.0)
    parser.add_argument("--kh-points", type=int, default=65)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--nice-level", type=int, default=0)
    parser.add_argument("--auto-jobs", action="store_true")
    parser.add_argument("--force-jobs", action="store_true")
    parser.add_argument("--resource-report", action="store_true")
    parser.add_argument("--benchmark-points", type=int, default=64)
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--production-confirm", action="store_true")
    return parser.parse_args()


def ensure_dirs() -> None:
    for path in (GRID_DIR, RAW_DIR, WORK_DIR, PROCESSED_DIR, PLOT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_constants(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fp:
        constants = json.load(fp)
    for bc in ("PBC", "OBC"):
        if bc not in constants:
            raise KeyError(f"{path} does not contain a {bc} entry")
        for key in ("g_pc", "nu", "y_h", "beta_over_nu"):
            if key not in constants[bc]:
                raise KeyError(f"{path} {bc} entry is missing {key}")
    return constants


def bc_name(pbc: int) -> str:
    return "PBC" if pbc else "OBC"


def bc_label(pbc: int) -> str:
    return "pbc" if pbc else "obc"


def method_for_L(L: int) -> str:
    return "ED" if L <= 12 else "Lanczos"


def fmt_float(value: float) -> str:
    return format(float(value), ".17e")


def linspace(start: float, stop: float, n: int) -> list[float]:
    if n <= 0:
        raise ValueError("grid point counts must be positive")
    if n == 1:
        return [float(start)]
    step = (float(stop) - float(start)) / float(n - 1)
    return [float(start) + i * step for i in range(n)]


def build_physical_grid(args: argparse.Namespace, c: dict, label: str) -> GridSpec:
    g_values = linspace(args.g_min, args.g_max, args.g_points)
    h_values = linspace(args.h_min, args.h_max, args.h_points)
    rows: list[tuple[float, float, float, float]] = []
    for h in h_values:
        for g in g_values:
            kg = (g - c["g_pc"]) * (args.L ** (1.0 / c["nu"]))
            kh = h * (args.L ** c["y_h"])
            rows.append((g, h, kg, kh))
    return GridSpec(
        "physical",
        GRID_DIR / f"grid_physical_{label}_L{args.L:02d}.dat",
        RAW_DIR / f"ghsurf_physical_{label}_L{args.L:02d}.dat",
        rows,
    )


def build_scaling_grid(args: argparse.Namespace, c: dict, label: str) -> GridSpec:
    kg_values = linspace(args.kg_min, args.kg_max, args.kg_points)
    kh_values = linspace(args.kh_min, args.kh_max, args.kh_points)
    rows: list[tuple[float, float, float, float]] = []
    for kh in kh_values:
        for kg in kg_values:
            g = c["g_pc"] + kg * (args.L ** (-1.0 / c["nu"]))
            h = kh * (args.L ** (-c["y_h"]))
            rows.append((g, h, kg, kh))
    return GridSpec(
        "scaling",
        GRID_DIR / f"grid_scaling_{label}_L{args.L:02d}.dat",
        RAW_DIR / f"ghsurf_scaling_{label}_L{args.L:02d}.dat",
        rows,
    )


def build_specs(args: argparse.Namespace, constants: dict) -> list[GridSpec]:
    label = bc_label(args.pbc)
    c = {
        key: float(value)
        for key, value in constants[bc_name(args.pbc)].items()
        if isinstance(value, (int, float))
    }
    specs: list[GridSpec] = []
    if args.grid_type in ("physical", "both"):
        specs.append(build_physical_grid(args, c, label))
    if args.grid_type in ("scaling", "both"):
        specs.append(build_scaling_grid(args, c, label))
    return specs


def write_grid(path: Path, rows: list[tuple[float, float, float, float]], meta: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fp:
        fp.write("# 1D Quantum Ising gh-surface grid\n")
        for key, value in meta.items():
            fp.write(f"# {key} = {value}\n")
        fp.write("# columns = g h kappa_g kappa_h\n")
        for g, h, kg, kh in rows:
            fp.write(f"{fmt_float(g)} {fmt_float(h)} {fmt_float(kg)} {fmt_float(kh)}\n")
        fp.flush()
        os.fsync(fp.fileno())
    tmp.replace(path)


def parse_meminfo() -> dict[str, int]:
    info: dict[str, int] = {}
    try:
        with Path("/proc/meminfo").open("r", encoding="utf-8") as fp:
            for line in fp:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(":")] = int(parts[1])
    except OSError:
        pass
    return info


def cpu_model() -> str:
    try:
        with Path("/proc/cpuinfo").open("r", encoding="utf-8") as fp:
            for line in fp:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def kib_to_gib(value: int | None) -> float | None:
    if value is None:
        return None
    return value / (1024.0 * 1024.0)


def resource_audit(L: int) -> ResourceInfo:
    mem = parse_meminfo()
    logical = os.cpu_count() or 1
    load_avg = os.getloadavg() if hasattr(os, "getloadavg") else None
    disk = shutil.disk_usage(PROJECT_ROOT)
    mem_available = kib_to_gib(mem.get("MemAvailable"))
    swap_total = kib_to_gib(mem.get("SwapTotal"))
    swap_free = kib_to_gib(mem.get("SwapFree"))
    swap_used = None
    if swap_total is not None and swap_free is not None:
        swap_used = max(0.0, swap_total - swap_free)

    jobs = 1
    reason_parts: list[str] = []
    if L <= 10:
        jobs = min(8, max(1, logical - 2))
        reason_parts.append("L<=10 ED policy")
    elif L == 12:
        jobs = min(4, max(1, logical // 2))
        reason_parts.append("L=12 dense ED policy")
    else:
        jobs = min(2, max(1, logical // 4))
        reason_parts.append("L>=14 Lanczos policy")

    if mem_available is not None and mem_available < 6.0 and L <= 10:
        jobs = min(jobs, 2)
        reason_parts.append("MemAvailable<6 GiB")
    if mem_available is not None and mem_available < 4.0:
        jobs = 1
        reason_parts.append("MemAvailable<4 GiB")
    if swap_used is not None and swap_used > 0.5:
        jobs = max(1, min(jobs, jobs // 2 or 1))
        reason_parts.append("swap is in use")

    return ResourceInfo(
        hostname=socket.gethostname(),
        cpu_model=cpu_model(),
        logical_cores=logical,
        load_avg=load_avg,
        mem_total_gib=kib_to_gib(mem.get("MemTotal")),
        mem_available_gib=mem_available,
        swap_total_gib=swap_total,
        swap_used_gib=swap_used,
        disk_available_gib=disk.free / (1024.0**3),
        thread_env={name: os.environ.get(name) for name in THREAD_ENV_VARS},
        recommended_jobs=max(1, jobs),
        reason=", ".join(reason_parts),
    )


def format_gib(value: float | None) -> str:
    return "unknown" if value is None else f"{value:.2f} GiB"


def print_resource_report(info: ResourceInfo, total_points: int, benchmark_avg: float | None = None) -> None:
    print("Resource report")
    print(f"  hostname                 = {info.hostname}")
    print(f"  cpu_model                = {info.cpu_model}")
    print(f"  logical_cores            = {info.logical_cores}")
    if info.load_avg is None:
        print("  load_average             = unknown")
    else:
        print(
            "  load_average             = "
            f"{info.load_avg[0]:.2f} {info.load_avg[1]:.2f} {info.load_avg[2]:.2f}"
        )
    print(f"  RAM total                = {format_gib(info.mem_total_gib)}")
    print(f"  RAM available            = {format_gib(info.mem_available_gib)}")
    print(f"  swap total               = {format_gib(info.swap_total_gib)}")
    print(f"  swap used                = {format_gib(info.swap_used_gib)}")
    print(f"  project disk available   = {info.disk_available_gib:.2f} GiB")
    print("  thread environment")
    for name in THREAD_ENV_VARS:
        print(f"    {name:24s} = {info.thread_env[name] or '<unset>'}")
    print(f"  recommended_jobs         = {info.recommended_jobs}")
    print(f"  reason                   = {info.reason}")
    print(f"  estimated_total_points   = {total_points}")
    if benchmark_avg is not None:
        print(f"  benchmark_avg_seconds_per_point = {benchmark_avg:.6g}")
        print(f"  estimated_wall_time jobs=1 = {format_seconds(benchmark_avg * total_points)}")
        print(
            f"  estimated_wall_time jobs={info.recommended_jobs} = "
            f"{format_seconds(benchmark_avg * total_points / max(1, info.recommended_jobs))}"
        )


def apply_auto_jobs(args: argparse.Namespace, info: ResourceInfo) -> int:
    requested = int(args.jobs)
    jobs_explicit = "--jobs" in sys.argv
    if args.auto_jobs and not jobs_explicit:
        return info.recommended_jobs
    if args.auto_jobs and jobs_explicit and requested > info.recommended_jobs and not args.force_jobs:
        print(
            f"WARN requested --jobs {requested} exceeds recommended_jobs "
            f"{info.recommended_jobs}; using recommendation. Pass --force-jobs to override."
        )
        return info.recommended_jobs
    return max(1, requested)


def subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    for name in THREAD_ENV_VARS:
        env[name] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def command_for(grid_type: str, grid_path: Path, out_path: Path, args: argparse.Namespace, *, resume: bool, overwrite: bool) -> list[str]:
    cmd = [
        "./ising_gh_surface",
        "--L",
        str(args.L),
        "--pbc",
        str(args.pbc),
        "--grid-type",
        grid_type,
        "--grid",
        str(grid_path.relative_to(PROJECT_ROOT)),
        "--out",
        str(out_path.relative_to(PROJECT_ROOT)),
        "--seed",
        str(args.seed),
    ]
    if resume:
        cmd.append("--resume")
    if overwrite:
        cmd.append("--overwrite")
    return cmd


def maybe_nice(cmd: list[str], nice_level: int) -> list[str]:
    if nice_level == 0:
        return cmd
    return ["nice", "-n", str(nice_level), *cmd]


def run_command_to_log(cmd: list[str], log_path: Path, nice_level: int) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    run_cmd = maybe_nice(cmd, nice_level)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + shlex.join(run_cmd) + "\n")
        log.flush()
        proc = subprocess.run(
            run_cmd,
            cwd=PROJECT_ROOT,
            env=subprocess_env(),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        tail = tail_text(log_path)
        raise RuntimeError(
            f"command failed rc={proc.returncode}: {shlex.join(run_cmd)}\n"
            f"log: {log_path.relative_to(PROJECT_ROOT)}\n{tail}"
        )


def tail_text(path: Path, n: int = 40) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])


def rows_key(row: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    return tuple(int(round(v / GRID_TOL)) for v in row)


def select_representative_rows(rows: list[tuple[float, float, float, float]], n: int) -> list[tuple[float, float, float, float]]:
    if n >= len(rows):
        return rows[:]
    wanted: dict[tuple[int, int, int, int], tuple[float, float, float, float]] = {}
    g_values = sorted({r[0] for r in rows})
    h_values = sorted({r[1] for r in rows})
    kg_values = sorted({r[2] for r in rows})
    kh_values = sorted({r[3] for r in rows})

    targets = [
        (g_values[0], h_values[0]),
        (g_values[-1], h_values[0]),
        (g_values[0], h_values[-1]),
        (g_values[-1], h_values[-1]),
        (min(g_values, key=abs), min(h_values, key=abs)),
        (min(g_values, key=lambda x: abs(x - 1.0)), min(h_values, key=abs)),
        (min(kg_values, key=abs), min(kh_values, key=abs)),
    ]
    for a, b in targets:
        for row in rows:
            if (abs(row[0] - a) <= GRID_TOL and abs(row[1] - b) <= GRID_TOL) or (
                abs(row[2] - a) <= GRID_TOL and abs(row[3] - b) <= GRID_TOL
            ):
                wanted[rows_key(row)] = row
                break

    if len(wanted) < n:
        step = max(1, (len(rows) - 1) // max(1, n - 1))
        for i in range(0, len(rows), step):
            wanted[rows_key(rows[i])] = rows[i]
            if len(wanted) >= n:
                break
    if len(wanted) < n:
        for row in rows:
            wanted[rows_key(row)] = row
            if len(wanted) >= n:
                break
    return list(wanted.values())[:n]


def format_seconds(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "unknown"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.1f}min"
    hours = minutes / 60.0
    if hours < 48:
        return f"{hours:.2f}h"
    return f"{hours / 24.0:.2f}d"


def run_benchmark(args: argparse.Namespace, specs: list[GridSpec], run_id: str, jobs: int, info: ResourceInfo) -> float:
    bench_dir = WORK_DIR / run_id / "benchmark"
    bench_dir.mkdir(parents=True, exist_ok=True)
    per_spec = max(1, args.benchmark_points // max(1, len(specs)))
    total_points = 0
    elapsed_total = 0.0
    print("Benchmark")
    for spec in specs:
        rows = select_representative_rows(spec.rows, per_spec)
        total_points += len(rows)
        grid_path = bench_dir / f"benchmark_grid_{spec.grid_type}.dat"
        out_path = bench_dir / f"benchmark_{spec.grid_type}.dat"
        log_path = bench_dir / f"benchmark_{spec.grid_type}.log"
        write_grid(
            grid_path,
            rows,
            {
                "grid_type": spec.grid_type,
                "L": str(args.L),
                "pbc": str(args.pbc),
                "BC": bc_name(args.pbc),
                "N_points": str(len(rows)),
                "run_id": run_id,
                "kind": "benchmark",
            },
        )
        cmd = command_for(spec.grid_type, grid_path, out_path, args, resume=False, overwrite=True)
        print(f"  run benchmark {spec.grid_type}: points={len(rows)}")
        t0 = time.perf_counter()
        run_command_to_log(cmd, log_path, args.nice_level)
        elapsed = time.perf_counter() - t0
        elapsed_total += elapsed
        print(f"    elapsed={elapsed:.3f}s avg={elapsed / len(rows):.6g}s/point")
    avg = elapsed_total / max(1, total_points)
    summary = {
        "run_id": run_id,
        "L": args.L,
        "pbc": args.pbc,
        "grid_type": args.grid_type,
        "benchmark_points": total_points,
        "avg_seconds_per_point": avg,
        "recommended_jobs": info.recommended_jobs,
        "actual_jobs_for_estimates": jobs,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(bench_dir / "benchmark_summary.json", summary)
    requested_points = sum(spec.n_points for spec in specs)
    print(f"  benchmark_points = {total_points}")
    print(f"  benchmark_avg_seconds_per_point = {avg:.6g}")
    print(f"  requested_grid_estimate jobs=1 = {format_seconds(avg * requested_points)}")
    print(f"  requested_grid_estimate jobs={info.recommended_jobs} = {format_seconds(avg * requested_points / max(1, info.recommended_jobs))}")
    return avg


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True)
        fp.write("\n")
        fp.flush()
        os.fsync(fp.fileno())
    tmp.replace(path)


def run_key(args: argparse.Namespace) -> dict:
    return {
        "L": args.L,
        "pbc": args.pbc,
        "grid_type": args.grid_type,
        "g_min": args.g_min,
        "g_max": args.g_max,
        "g_points": args.g_points,
        "h_min": args.h_min,
        "h_max": args.h_max,
        "h_points": args.h_points,
        "kg_min": args.kg_min,
        "kg_max": args.kg_max,
        "kg_points": args.kg_points,
        "kh_min": args.kh_min,
        "kh_max": args.kh_max,
        "kh_points": args.kh_points,
        "seed": args.seed,
    }


def metadata_matches(meta: dict, key: dict) -> bool:
    return all(meta.get(k) == v for k, v in key.items())


def latest_matching_run_id(args: argparse.Namespace) -> str | None:
    key = run_key(args)
    if not WORK_DIR.exists():
        return None
    matches: list[tuple[int, str]] = []
    for path in WORK_DIR.iterdir():
        meta_path = path / "run_meta.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if metadata_matches(meta.get("run_key", {}), key):
            matches.append((meta_path.stat().st_mtime_ns, path.name))
    if not matches:
        return None
    matches.sort(reverse=True)
    return matches[0][1]


def choose_run_id(args: argparse.Namespace) -> str:
    if args.run_id:
        return args.run_id
    if args.resume or args.merge_only:
        found = latest_matching_run_id(args)
        if found:
            return found
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def chunk_rows(rows: list[tuple[float, float, float, float]], jobs: int, chunk_size: int | None) -> list[list[tuple[float, float, float, float]]]:
    if not rows:
        return []
    if chunk_size is None:
        if len(rows) <= jobs * 50:
            chunk_size = max(1, math.ceil(len(rows) / max(1, jobs)))
        else:
            target_shards = max(jobs, min(len(rows), jobs * 4))
            chunk_size = max(1, math.ceil(len(rows) / target_shards))
    if chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    return [rows[i : i + chunk_size] for i in range(0, len(rows), chunk_size)]


def make_shards(spec: GridSpec, args: argparse.Namespace, run_id: str, jobs: int) -> list[ShardSpec]:
    grid_dir = WORK_DIR / run_id / "grids"
    raw_dir = WORK_DIR / run_id / "raw_shards"
    log_dir = WORK_DIR / run_id / "logs"
    chunks = chunk_rows(spec.rows, jobs, args.chunk_size)
    shards: list[ShardSpec] = []
    for i, rows in enumerate(chunks):
        tag = f"{spec.grid_type}_{bc_label(args.pbc)}_L{args.L:02d}_shard_{i:04d}"
        shards.append(
            ShardSpec(
                spec.grid_type,
                i,
                grid_dir / f"grid_{tag}.dat",
                raw_dir / f"ghsurf_{tag}.dat",
                log_dir / f"{tag}.log",
                rows,
            )
        )
    return shards


def count_valid_rows(path: Path) -> tuple[int, int]:
    rows = 0
    bad = 0
    if not path.exists():
        return (0, 0)
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) != len(DATA_COLUMNS):
                bad += 1
            else:
                rows += 1
    return rows, bad


def output_complete(path: Path, expected_rows: int) -> bool:
    rows, bad = count_valid_rows(path)
    return bad == 0 and rows == expected_rows


def run_shards(args: argparse.Namespace, specs: list[GridSpec], run_id: str, jobs: int) -> dict[str, list[ShardSpec]]:
    if not EXECUTABLE.exists():
        raise FileNotFoundError(f"{EXECUTABLE.relative_to(PROJECT_ROOT)} is missing; run `make ghsurface` first")

    shard_map: dict[str, list[ShardSpec]] = {}
    for spec in specs:
        write_grid(
            spec.grid_path,
            spec.rows,
            {
                "grid_type": spec.grid_type,
                "L": str(args.L),
                "pbc": str(args.pbc),
                "BC": bc_name(args.pbc),
                "N_points": str(spec.n_points),
                "run_id": run_id,
            },
        )
        shards = make_shards(spec, args, run_id, jobs)
        shard_map[spec.grid_type] = shards
        for shard in shards:
            shard.out_path.parent.mkdir(parents=True, exist_ok=True)
            shard.log_path.parent.mkdir(parents=True, exist_ok=True)
            write_grid(
                shard.grid_path,
                shard.rows,
                {
                    "grid_type": shard.grid_type,
                    "L": str(args.L),
                    "pbc": str(args.pbc),
                    "BC": bc_name(args.pbc),
                    "N_points": str(shard.n_points),
                    "run_id": run_id,
                    "shard_index": str(shard.index),
                },
            )

    tasks: list[tuple[ShardSpec, list[str]]] = []
    skipped = 0
    for shards in shard_map.values():
        for shard in shards:
            if args.resume and output_complete(shard.out_path, shard.n_points):
                skipped += 1
                continue
            resume = args.resume and shard.out_path.exists()
            overwrite = args.overwrite and not resume
            cmd = command_for(shard.grid_type, shard.grid_path, shard.out_path, args, resume=resume, overwrite=overwrite)
            tasks.append((shard, cmd))

    print(f"Shard execution: jobs={jobs} shards_total={sum(len(v) for v in shard_map.values())} skipped_complete={skipped} to_run={len(tasks)}")
    if not tasks:
        return shard_map

    def one(task: tuple[ShardSpec, list[str]]) -> str:
        shard, cmd = task
        run_command_to_log(cmd, shard.log_path, args.nice_level)
        return f"{shard.grid_type} shard {shard.index:04d}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(one, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            print(f"  [OK] {future.result()}")
            sys.stdout.flush()
    return shard_map


def parse_data_row(line: str) -> tuple[tuple[float, float, float, float], list[str]]:
    parts = line.split()
    if len(parts) != len(DATA_COLUMNS):
        raise ValueError(f"expected {len(DATA_COLUMNS)} columns, found {len(parts)}")
    coords = tuple(float(parts[i]) for i in range(4))
    return coords, parts


def coord_key(row: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    return tuple(int(round(x / GRID_TOL)) if math.isfinite(x) else 999999999 for x in row)


def grid_ranges(rows: list[tuple[float, float, float, float]]) -> dict[str, float]:
    arr = list(zip(*rows))
    return {
        "g_min": min(arr[0]),
        "g_max": max(arr[0]),
        "h_min": min(arr[1]),
        "h_max": max(arr[1]),
        "kappa_g_min": min(arr[2]),
        "kappa_g_max": max(arr[2]),
        "kappa_h_min": min(arr[3]),
        "kappa_h_max": max(arr[3]),
    }


def merge_shards(spec: GridSpec, shards: list[ShardSpec], args: argparse.Namespace, run_id: str) -> dict:
    expected = {coord_key(row): i for i, row in enumerate(spec.rows)}
    if len(expected) != len(spec.rows):
        raise RuntimeError(f"global {spec.grid_type} grid has duplicate coordinates")

    seen: dict[tuple[int, int, int, int], list[str]] = {}
    duplicate_count = 0
    for shard in shards:
        if not output_complete(shard.out_path, shard.n_points):
            rows, bad = count_valid_rows(shard.out_path)
            raise RuntimeError(
                f"incomplete shard {shard.out_path.relative_to(PROJECT_ROOT)} rows={rows} bad={bad} expected={shard.n_points}"
            )
        with shard.out_path.open("r", encoding="utf-8") as fp:
            for line_no, line in enumerate(fp, start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                coords, parts = parse_data_row(stripped)
                key = coord_key(coords)
                if key not in expected:
                    raise RuntimeError(
                        f"row in {shard.out_path.relative_to(PROJECT_ROOT)}:{line_no} is not in global grid"
                    )
                if key in seen:
                    duplicate_count += 1
                seen[key] = parts

    missing = len(spec.rows) - len(seen)
    if duplicate_count or missing:
        raise RuntimeError(
            f"merge failed for {spec.grid_type}: duplicates={duplicate_count} missing={missing}"
        )

    ranges = grid_ranges(spec.rows)
    tmp = spec.out_path.with_suffix(spec.out_path.suffix + ".tmp")
    spec.out_path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as fp:
        fp.write("# 1D Quantum Ising -- gh surface merged output\n")
        fp.write(f"# merge_timestamp = {datetime.now().isoformat(timespec='seconds')}\n")
        fp.write(f"# grid_type = {spec.grid_type}\n")
        fp.write(f"# L = {args.L}\n")
        fp.write(f"# pbc = {args.pbc}\n")
        fp.write(f"# BC = {bc_name(args.pbc)}\n")
        fp.write(f"# seed = {args.seed}\n")
        fp.write(f"# N_points = {spec.n_points}\n")
        for key, value in ranges.items():
            fp.write(f"# {key} = {fmt_float(value)}\n")
        fp.write(f"# number_shards = {len(shards)}\n")
        fp.write(f"# run_id = {run_id}\n")
        fp.write(f"# source_shard_directory = {(WORK_DIR / run_id / 'raw_shards').relative_to(PROJECT_ROOT)}\n")
        fp.write("# columns = " + " ".join(DATA_COLUMNS) + "\n")
        fp.write("# command = " + shlex.join(sys.argv) + "\n")
        fp.write("#\n")
        for row in spec.rows:
            fp.write(" ".join(seen[coord_key(row)]) + "\n")
        fp.flush()
        os.fsync(fp.fileno())
    tmp.replace(spec.out_path)
    return {
        "grid_type": spec.grid_type,
        "rows": spec.n_points,
        "columns": len(DATA_COLUMNS),
        "duplicates": duplicate_count,
        "missing": missing,
        "shards": len(shards),
        "output": str(spec.out_path.relative_to(PROJECT_ROOT)),
    }


def remove_outputs_for_overwrite(specs: list[GridSpec], run_dir: Path) -> None:
    for spec in specs:
        for path in (spec.out_path, spec.out_path.with_suffix(spec.out_path.suffix + ".ckpt"), Path(str(spec.out_path) + ".lock")):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    if run_dir.exists():
        shutil.rmtree(run_dir)


def print_summary(args: argparse.Namespace, constants: dict, specs: list[GridSpec], info: ResourceInfo, jobs: int, run_id: str) -> None:
    c = constants[bc_name(args.pbc)]
    print("GH surface driver")
    print(f"  L              = {args.L}")
    print(f"  BC             = {bc_name(args.pbc)}")
    print(f"  method         = {method_for_L(args.L)}")
    if args.L >= 14:
        print("  WARN           = L>=14 uses Lanczos; run benchmark before production")
    print(f"  executable     = {EXECUTABLE.relative_to(PROJECT_ROOT)}")
    print(f"  run_id         = {run_id}")
    print("  FSS constants")
    print(f"    g_pc         = {float(c['g_pc']):.17g}")
    print(f"    nu           = {float(c['nu']):.17g}")
    print(f"    y_h          = {float(c['y_h']):.17g}")
    print(f"    beta_over_nu = {float(c['beta_over_nu']):.17g}")
    print(f"  jobs           = {jobs}")
    print(f"  recommended    = {info.recommended_jobs} ({info.reason})")
    print(f"  dry_run        = {'yes' if args.dry_run else 'no'}")
    print(f"  resume         = {'yes' if args.resume else 'no'}")
    print(f"  overwrite      = {'yes' if args.overwrite else 'no'}")
    print("  grids")
    for spec in specs:
        shards = len(chunk_rows(spec.rows, jobs, args.chunk_size))
        print(f"    {spec.grid_type:8s}: points={spec.n_points} planned_shards={shards}")
        print(f"      grid = {spec.grid_path.relative_to(PROJECT_ROOT)}")
        print(f"      out  = {spec.out_path.relative_to(PROJECT_ROOT)}")
    print(f"  total tuples   = {sum(spec.n_points for spec in specs)}")


def dry_run(args: argparse.Namespace, specs: list[GridSpec], jobs: int, run_id: str) -> None:
    print("Dry-run shard plan")
    for spec in specs:
        shards = make_shards(spec, args, run_id, jobs)
        print(f"  {spec.grid_type}: final={spec.out_path.relative_to(PROJECT_ROOT)} shards={len(shards)}")
        for shard in shards[:6]:
            cmd = command_for(shard.grid_type, shard.grid_path, shard.out_path, args, resume=args.resume, overwrite=args.overwrite)
            print(f"    shard {shard.index:04d} points={shard.n_points} cmd={shlex.join(cmd)}")
        if len(shards) > 6:
            print(f"    ... {len(shards) - 6} more shards")


def validate_args(args: argparse.Namespace) -> None:
    if args.resume and args.overwrite and args.merge_only:
        raise SystemExit("--merge-only cannot combine --resume and --overwrite")
    if args.resume and args.overwrite and not args.production_confirm:
        raise SystemExit("--resume and --overwrite together require --production-confirm")
    for name in ("g_points", "h_points", "kg_points", "kh_points", "benchmark_points"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.chunk_size is not None and args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be positive")
    for name in ("g_min", "g_max", "h_min", "h_max", "kg_min", "kg_max", "kh_min", "kh_max"):
        if not math.isfinite(getattr(args, name)):
            raise SystemExit(f"--{name.replace('_', '-')} must be finite")


def main() -> None:
    args = parse_args()
    validate_args(args)
    ensure_dirs()

    constants = load_constants(CONSTANTS_PATH)
    specs = build_specs(args, constants)
    total_points = sum(spec.n_points for spec in specs)
    info = resource_audit(args.L)
    jobs = apply_auto_jobs(args, info)
    run_id = choose_run_id(args)
    run_dir = WORK_DIR / run_id

    print_summary(args, constants, specs, info, jobs, run_id)
    if args.auto_jobs or args.resource_report:
        print_resource_report(info, total_points)

    if args.resource_report:
        return

    if args.dry_run:
        dry_run(args, specs, jobs, run_id)
        return

    benchmark_avg: float | None = None
    if args.benchmark_only:
        benchmark_avg = run_benchmark(args, specs, run_id, jobs, info)
        print_resource_report(info, total_points, benchmark_avg)
        return

    if args.L >= 14:
        marker = run_dir / "benchmark" / "benchmark_summary.json"
        if not marker.exists():
            raise SystemExit(
                "L>=14 production requires a benchmark marker for this run_id. "
                "Run first with --benchmark-only --run-id SAME_ID."
            )

    if total_points > PRODUCTION_POINT_THRESHOLD and not args.production_confirm and not args.merge_only:
        print(
            f"Production confirmation required: total_points={total_points} exceeds "
            f"{PRODUCTION_POINT_THRESHOLD}."
        )
        print("Re-run with --production-confirm after dry-run/benchmark.")
        return

    if args.overwrite and not args.merge_only:
        remove_outputs_for_overwrite(specs, run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        run_dir / "run_meta.json",
        {
            "run_id": run_id,
            "run_key": run_key(args),
            "created_or_updated": datetime.now().isoformat(timespec="seconds"),
            "jobs": jobs,
            "recommended_jobs": info.recommended_jobs,
            "command": shlex.join(sys.argv),
        },
    )

    if args.merge_only:
        shard_map = {spec.grid_type: make_shards(spec, args, run_id, jobs) for spec in specs}
    else:
        for spec in specs:
            if spec.out_path.exists() and not (args.resume or args.overwrite):
                raise SystemExit(
                    f"output exists: {spec.out_path.relative_to(PROJECT_ROOT)}; "
                    "pass --resume or --overwrite"
                )
        shard_map = run_shards(args, specs, run_id, jobs)

    merge_results = []
    for spec in specs:
        result = merge_shards(spec, shard_map[spec.grid_type], args, run_id)
        merge_results.append(result)
        print(
            f"  [MERGE] {result['output']} rows={result['rows']} "
            f"shards={result['shards']} duplicates={result['duplicates']} missing={result['missing']}"
        )
    write_json(run_dir / "merge_summary.json", {"results": merge_results})
    print("done")


if __name__ == "__main__":
    main()
