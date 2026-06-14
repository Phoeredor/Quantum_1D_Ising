#!/usr/bin/env python3
"""Compatibility entry point for the Majorana report-figure script."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "majorana_modes.py"
    runpy.run_path(str(target), run_name="__main__")
