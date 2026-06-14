#!/usr/bin/env python3
"""Export h=0 FSS constants for the quench production driver."""

from __future__ import annotations

import argparse
import json
import shlex
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "h_null" / "fss" / "fss_constants.json"
DEFAULT_AUDIT = PROJECT_ROOT / "data" / "quench" / "quench_constants_used.json"


def load_json_decimal(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh, parse_float=Decimal, parse_int=Decimal)
    except FileNotFoundError as exc:
        raise SystemExit(f"[ERROR] FSS constants file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"[ERROR] cannot parse JSON {path}: {exc}") from exc


def as_decimal(value: Any, field: str, bc: str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, str):
        try:
            return Decimal(value)
        except InvalidOperation as exc:
            raise SystemExit(f"[ERROR] {bc}.{field} is not numeric: {value!r}") from exc
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    raise SystemExit(f"[ERROR] {bc}.{field} is missing or not numeric")


def pick(block: dict[str, Any], keys: tuple[str, ...], bc: str, label: str) -> Decimal:
    for key in keys:
        if key in block:
            return as_decimal(block[key], key, bc)
    key_list = ", ".join(keys)
    raise SystemExit(f"[ERROR] missing {label} for {bc}; tried keys: {key_list}")


def fmt(value: Decimal) -> str:
    return format(value, ".17g")


def extract_bc_constants(data: dict[str, Any], bc: str) -> tuple[dict[str, str], bool]:
    block = data.get(bc)
    if not isinstance(block, dict):
        raise SystemExit(f"[ERROR] missing top-level {bc} block in FSS constants")

    g_pc = pick(block, ("g_pc", "gpc", "g_c", "gc"), bc, "g_pc")
    beta_over_nu = pick(block, ("beta_over_nu", "beta_ov_nu", "beta_overnu"), bc, "beta_over_nu")
    z = pick(block, ("z", "z_dynamic"), bc, "z")

    if "nu" in block:
        nu = as_decimal(block["nu"], "nu", bc)
    else:
        inv_nu = pick(
            block,
            ("inv_nu", "inverse_nu", "one_over_nu", "invnu", "nu_inv"),
            bc,
            "nu or inverse nu",
        )
        if inv_nu == 0:
            raise SystemExit(f"[ERROR] {bc}.inv_nu is zero")
        nu = Decimal(1) / inv_nu

    derived_y_h = False
    if "y_h" in block:
        y_h = as_decimal(block["y_h"], "y_h", bc)
    else:
        y_h = Decimal(1) + z - beta_over_nu
        derived_y_h = True

    constants = {
        "g_pc": fmt(g_pc),
        "beta_over_nu": fmt(beta_over_nu),
        "nu": fmt(nu),
        "z": fmt(z),
        "y_h": fmt(y_h),
        "approach": str(block.get("approach", "")),
        "Lmin": fmt(as_decimal(block["Lmin"], "Lmin", bc)) if "Lmin" in block else "",
    }
    return constants, derived_y_h


def write_audit(
    audit_path: Path,
    source_path: Path,
    constants_by_bc: dict[str, dict[str, str]],
    derived_y_h: dict[str, bool],
) -> None:
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_file": str(source_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "constants_by_bc": constants_by_bc,
        "derived_y_h_if_applicable": derived_y_h,
        "y_h_note": "If y_h is absent in the source JSON, y_h is derived as 1 + z - beta_over_nu with d=1; otherwise the source y_h value is used.",
    }
    with audit_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


def shell_assign(name: str, value: str) -> str:
    return f"{name}={shlex.quote(value)}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export quench constants from h=0 FSS JSON.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()

    source = args.input
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    audit = args.audit
    if not audit.is_absolute():
        audit = PROJECT_ROOT / audit

    data = load_json_decimal(source)
    pbc, pbc_derived = extract_bc_constants(data, "PBC")
    obc, obc_derived = extract_bc_constants(data, "OBC")
    constants_by_bc = {"PBC": pbc, "OBC": obc}
    derived_y_h = {"PBC": pbc_derived, "OBC": obc_derived}

    write_audit(audit, source, constants_by_bc, derived_y_h)

    print(shell_assign("QUENCH_CONSTANTS_SOURCE", str(source)))
    for prefix, constants in (("PBC", pbc), ("OBC", obc)):
        print(shell_assign(f"{prefix}_GPC", constants["g_pc"]))
        print(shell_assign(f"{prefix}_BETA_OVER_NU", constants["beta_over_nu"]))
        print(shell_assign(f"{prefix}_NU", constants["nu"]))
        print(shell_assign(f"{prefix}_Z", constants["z"]))
        print(shell_assign(f"{prefix}_Y_H", constants["y_h"]))


if __name__ == "__main__":
    main()
