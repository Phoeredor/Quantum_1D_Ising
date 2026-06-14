#!/usr/bin/env python3
"""Export finite-size-scaling constants for h-field production runs.

The exporter reads the deterministic FSS sweep tables produced by
scripts/h_null/fss_h_null_analysis.py and writes
data/h_null/fss/fss_constants.json.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FSS_DIR = PROJECT_ROOT / "data" / "h_null" / "fss"
EXPONENT_SWEEPS = FSS_DIR / "exponent_sweeps.dat"
BETA_SWEEP = FSS_DIR / "beta_over_nu_sweep.dat"
OUTPUT = FSS_DIR / "fss_constants.json"
D_SPATIAL = 1.0


def _as_float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return math.nan


def _read_exponent_sweeps(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 13:
                continue
            rows.append(
                {
                    "BC": parts[0],
                    "quantity": parts[1],
                    "approach": parts[2],
                    "Lmin": int(parts[3]),
                    "g_pc": _as_float(parts[4]),
                    "gpc_source": parts[5],
                    "value": _as_float(parts[6]),
                    "lmin_drift_final": _as_float(parts[7]),
                    "omega": _as_float(parts[8]),
                    "Le_shift": _as_float(parts[9]),
                    "B": _as_float(parts[10]),
                    "resid_rms": _as_float(parts[11]),
                    "n": int(float(parts[12])),
                }
            )
    return rows


def _read_beta_sweep(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 14:
                continue
            rows.append(
                {
                    "BC": parts[0],
                    "observable": parts[1],
                    "role": parts[2],
                    "approach": parts[3],
                    "Lmin": int(parts[4]),
                    "g_pc": _as_float(parts[5]),
                    "gpc_source": parts[6],
                    "value": _as_float(parts[7]),
                    "B": _as_float(parts[8]),
                    "omega": _as_float(parts[9]),
                    "Le_shift": _as_float(parts[10]),
                    "resid_rms": _as_float(parts[11]),
                    "n": int(float(parts[12])),
                    "is_final_approach": bool(int(parts[13])),
                    "is_final_lmin": bool(int(parts[14])) if len(parts) > 14 else False,
                }
            )
    return rows


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


def _max_drift(rows: list[dict], final_lmin: int, key: str, final_value: float) -> float:
    values = [
        abs(float(row[key]) - final_value)
        for row in rows
        if int(row["Lmin"]) >= int(final_lmin) and _finite(row.get(key, math.nan))
    ]
    return max(values) if values and _finite(final_value) else math.nan


def _index_exponent(rows: list[dict], bc: str, quantity: str, approach: str) -> dict[int, dict]:
    return {
        int(row["Lmin"]): row
        for row in rows
        if row["BC"] == bc and row["quantity"] == quantity and row["approach"] == approach
    }


def _index_beta(rows: list[dict], bc: str, approach: str) -> dict[int, dict]:
    return {
        int(row["Lmin"]): row
        for row in rows
        if row["BC"] == bc
        and row["observable"] == "psi_tilde"
        and row["role"] == "primary"
        and row["approach"] == approach
    }


def _select_final(beta_rows: list[dict], bc: str) -> tuple[str, int]:
    final = [
        row
        for row in beta_rows
        if row["BC"] == bc
        and row["observable"] == "psi_tilde"
        and row["role"] == "primary"
        and row["is_final_approach"]
        and row["is_final_lmin"]
    ]
    if final:
        row = final[0]
        return str(row["approach"]), int(row["Lmin"])

    fallback = "subleading" if bc == "PBC" else "mixed_subleading"
    print(
        f"[WARN] could not find final flags for {bc}; "
        f"falling back to approach={fallback}, Lmin=10",
        file=sys.stderr,
    )
    return fallback, 10


def _build_bc_constants(
    bc: str,
    exponent_rows: list[dict],
    beta_rows: list[dict],
) -> dict:
    approach, final_lmin = _select_final(beta_rows, bc)
    nu_rows = _index_exponent(exponent_rows, bc, "nu_inv", approach)
    z_rows = _index_exponent(exponent_rows, bc, "z_dynamic", approach)
    beta_index = _index_beta(beta_rows, bc, approach)

    common_lmins = sorted(set(z_rows) & set(beta_index))
    derived_sweep: list[dict] = []
    full_sweep: list[dict] = []

    for Lmin in common_lmins:
        z = float(z_rows[Lmin]["value"])
        beta_over_nu = float(beta_index[Lmin]["value"])
        gamma_over_nu = D_SPATIAL + z - 2.0 * beta_over_nu
        eta = 2.0 - gamma_over_nu
        y_h = 0.5 * (D_SPATIAL + z + 2.0 - eta)
        y_h_check = beta_over_nu + gamma_over_nu
        if _finite(y_h) and _finite(y_h_check) and abs(y_h - y_h_check) > 1e-10:
            print(
                f"[WARN] {bc} Lmin={Lmin}: y_h consistency check differs "
                f"({y_h:.12e} vs {y_h_check:.12e})",
                file=sys.stderr,
            )
        delta = y_h / beta_over_nu if beta_over_nu != 0.0 else math.nan
        derived_sweep.append(
            {
                "Lmin": Lmin,
                "eta": eta,
                "y_h": y_h,
                "delta": delta,
            }
        )
        full_sweep.append(
            {
                "Lmin": Lmin,
                "z": z,
                "beta_over_nu": beta_over_nu,
                "gamma_over_nu": gamma_over_nu,
                "eta": eta,
                "y_h": y_h,
                "delta": delta,
            }
        )

    final = next((row for row in full_sweep if row["Lmin"] == final_lmin), None)
    if final is None:
        raise RuntimeError(
            f"missing derived final row for {bc} approach={approach} Lmin={final_lmin}"
        )

    nu_row = nu_rows.get(final_lmin)
    beta_row = beta_index.get(final_lmin)
    if nu_row is None or beta_row is None:
        raise RuntimeError(
            f"missing primary final row for {bc} approach={approach} Lmin={final_lmin}"
        )

    inv_nu = float(nu_row["value"])
    nu = 1.0 / inv_nu if inv_nu != 0.0 else math.nan
    eta_drift = _max_drift(derived_sweep, final_lmin, "eta", final["eta"])
    yh_drift = _max_drift(derived_sweep, final_lmin, "y_h", final["y_h"])
    delta_drift = _max_drift(derived_sweep, final_lmin, "delta", final["delta"])

    if not _finite(eta_drift):
        print(f"[WARN] eta Lmin drift unavailable for {bc}", file=sys.stderr)
    if not _finite(yh_drift):
        print(f"[WARN] y_h Lmin drift unavailable for {bc}", file=sys.stderr)
    if not _finite(delta_drift):
        print(f"[WARN] delta Lmin drift unavailable for {bc}", file=sys.stderr)

    return {
        "g_pc": float(beta_row["g_pc"]),
        "z": final["z"],
        "inv_nu": inv_nu,
        "nu": nu,
        "beta_over_nu": final["beta_over_nu"],
        "gamma_over_nu": final["gamma_over_nu"],
        "eta": final["eta"],
        "y_h": final["y_h"],
        "delta": final["delta"],
        "Lmin": final_lmin,
        "approach": approach,
        "derived_sweep": derived_sweep,
        "eta_lmin_drift": eta_drift,
        "y_h_lmin_drift": yh_drift,
        "delta_lmin_drift": delta_drift,
    }


def main() -> int:
    FSS_DIR.mkdir(parents=True, exist_ok=True)
    exponent_rows = _read_exponent_sweeps(EXPONENT_SWEEPS)
    beta_rows = _read_beta_sweep(BETA_SWEEP)

    payload = {
        "PBC": _build_bc_constants("PBC", exponent_rows, beta_rows),
        "OBC": _build_bc_constants("OBC", exponent_rows, beta_rows),
    }

    with OUTPUT.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True, allow_nan=True)
        fp.write("\n")

    print(f"[OK] wrote {OUTPUT.relative_to(PROJECT_ROOT)}")
    for bc in ("PBC", "OBC"):
        item = payload[bc]
        print(
            f"  {bc}: g_pc={item['g_pc']:.8f} y_h={item['y_h']:.12g} "
            f"eta={item['eta']:.12g} delta={item['delta']:.12g} "
            f"Lmin={item['Lmin']} approach={item['approach']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
