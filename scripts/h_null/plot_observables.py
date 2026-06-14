#!/usr/bin/env python3
"""
Observable diagnostic plots for deterministic ED/Lanczos data of the
one-dimensional quantum Ising chain.

Inputs are read from data/h_null/observables and data/h_null/chiz_fd.
Figures are written to plots/h_null/observables.

Usage: python3 scripts/h_null/plot_observables.py
"""

import os
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    mpl_cache_dir = Path("/tmp/qising_1d_matplotlib_cache/h_null")
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache_dir)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import brentq

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OBSERVABLES_DIR = PROJECT_ROOT / "data" / "h_null" / "observables"
CHIZ_FD_DIR = PROJECT_ROOT / "data" / "h_null" / "chiz_fd"
FSS_DIR = PROJECT_ROOT / "data" / "h_null" / "fss"
PLOT_DIR = PROJECT_ROOT / "plots" / "h_null" / "observables"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
NUMERIC_OUTPUT_SUFFIXES = {".dat", ".txt", ".csv", ".json", ".npy", ".npz"}

ED_SIZES = [4, 6, 8, 10, 12]
LZ_SIZES = [14, 16, 18, 20, 22]
ALL_SIZES = ED_SIZES + LZ_SIZES
PBC_SIZES = ALL_SIZES
OBC_SIZES = ALL_SIZES
CHIZ_PBC_EXTRA_SIZES = []
CHIZ_PBC_SIZES = PBC_SIZES + CHIZ_PBC_EXTRA_SIZES
CHIZ_OBC_SIZES = OBC_SIZES

_cmap_pbc = plt.cm.plasma
_cmap_obc = plt.cm.viridis
_color_values = np.linspace(0.0, 0.85, len(ALL_SIZES))
COLORS_PBC = {L: _cmap_pbc(v) for L, v in zip(ALL_SIZES, _color_values)}
COLORS_OBC = {L: _cmap_obc(v) for L, v in zip(ALL_SIZES, _color_values)}
PBC_ACCENT = _cmap_pbc(0.65)
OBC_ACCENT = _cmap_obc(0.65)
MX_THEORY_COLORS = {"PBC": "#00A6A6", "OBC": "#CC0077"}

_extra_color_values = np.linspace(0.88, 0.98, len(CHIZ_PBC_EXTRA_SIZES))
for L, v in zip(CHIZ_PBC_EXTRA_SIZES, _extra_color_values):
    if L not in COLORS_PBC:
        COLORS_PBC[L] = _cmap_pbc(v)


def colors_for_bc(bc_label: str) -> dict[int, tuple]:
    return COLORS_PBC if bc_label.upper() == "PBC" else COLORS_OBC


def snapshot_outputs() -> dict[Path, tuple[int, int]]:
    """Capture plot output metadata before generation."""
    snapshot: dict[Path, tuple[int, int]] = {}
    for path in PLOT_DIR.glob("*"):
        if path.is_file():
            st = path.stat()
            snapshot[path] = (st.st_size, st.st_mtime_ns)
    return snapshot


def report_output_changes(before: dict[Path, tuple[int, int]]) -> None:
    """Print a compact before/after report for files touched by this script."""
    changed: list[tuple[Path, str, int | None, int]] = []
    for path in sorted(PLOT_DIR.glob("*")):
        if not path.is_file():
            continue
        st = path.stat()
        after = (st.st_size, st.st_mtime_ns)
        old = before.get(path)
        if old is None:
            changed.append((path, "created", None, st.st_size))
        elif old != after:
            changed.append((path, "updated", old[0], st.st_size))

    numeric = [item for item in changed if item[0].suffix.lower() in NUMERIC_OUTPUT_SUFFIXES]
    pdfs = [item for item in changed if item[0].suffix.lower() == ".pdf"]

    print("\n[plot_observables.py]  Output before/after summary:")
    if numeric:
        print("  Numeric outputs changed:")
        for path, status, old_size, new_size in numeric:
            if old_size is None:
                print(f"    {status:7s} {path.relative_to(PROJECT_ROOT)} size={new_size} B")
            else:
                print(
                    f"    {status:7s} {path.relative_to(PROJECT_ROOT)} "
                    f"size={old_size} -> {new_size} B"
                )
    else:
        print("  Numeric outputs changed: none")

    if pdfs:
        print("  PDF outputs changed:")
        for path, status, old_size, new_size in pdfs:
            if old_size is None:
                print(f"    {status:7s} {path.name} size={new_size} B")
            else:
                print(f"    {status:7s} {path.name} size={old_size} -> {new_size} B")

FONT_SCALE = 1.6
AXIS_LABEL_FONTSIZE = 18
TICK_FONTSIZE = 15
LEGEND_FONTSIZE = 15
TITLE_FONTSIZE = 15
TEXT_FONTSIZE = 15
INSET_TICK_FONTSIZE = 12
EXTERNAL_LEGEND_Y_ANCHOR = -0.15

# Observable column indices in obs files.
COL = dict(g=0, Mx=1, mz_sq=2, mz=3, chi_z=4,
           mz4=5, psi_tilde=6, psi_bar=7,
           binder=8, chi_perp=9, g_chi_perp=10)
CHIZFD_COL = dict(g=0, dh=1, method_code=2, chi_fd=7, oddness1=8, oddness2=9)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def gap_columns(gd: np.ndarray) -> dict:
    gd = np.atleast_2d(gd)
    nc = gd.shape[1]
    # ED: 8 cols -> gap=col5, gapL=col6, E2=col3
    # LZ: 7 cols -> gap=col4, gapL=col5, E2=col3
    gap_col = 5 if nc == 8 else 4
    gapL_col = 6 if nc == 8 else 5
    # E0=col1, E1=col2, E2=col3 -> same in both formats
    g_arr = gd[:, 0]
    E0 = gd[:, 1]
    Delta0 = gd[:, gap_col]
    Delta1 = gd[:, 3] - gd[:, 1]  # E2 - E0
    return {
        "nc": nc,
        "gap_col": gap_col,
        "gapL_col": gapL_col,
        "g_arr": g_arr,
        "E0": E0,
        "Delta0": Delta0,
        "Delta1": Delta1,
    }


def data_dir_for_bc(pbc: bool) -> Path:
    return OBSERVABLES_DIR / ("PBC" if pbc else "OBC")


def load_data(pbc: bool = True) -> dict:
    """
    Load all available ED and Lanczos data for one boundary condition.
    pbc=True  -> gap_L*.dat / obs_L*.dat and gap_lz_L*.dat / obs_lz_L*.dat
    pbc=False -> gap_obc_L*.dat / obs_obc_L*.dat and
                 gap_lz_obc_L*.dat / obs_lz_obc_L*.dat
    Returns {L: (gap_array, obs_array, source)} where source is 'ED' or 'LNCZ'.
    """
    data: dict = {}
    suffix = "" if pbc else "_obc"
    sizes = PBC_SIZES if pbc else OBC_SIZES
    data_dir = data_dir_for_bc(pbc)
    for L in sizes:
        if L in ED_SIZES:
            gf = data_dir / f"gap{suffix}_L{L:02d}.dat"
            of = data_dir / f"obs{suffix}_L{L:02d}.dat"
            src = "ED"
        else:
            gf = data_dir / f"gap_lz{suffix}_L{L:02d}.dat"
            of = data_dir / f"obs_lz{suffix}_L{L:02d}.dat"
            src = "LNCZ"
        if gf.exists() and of.exists():
            try:
                gd = np.atleast_2d(np.loadtxt(gf, comments="#"))
                _ = gap_columns(gd)
                od = np.atleast_2d(np.loadtxt(of, comments="#"))
                # Drop rows with non-finite mandatory observables.
                gd = gd[np.isfinite(gd).all(axis=1)]
                # psi_tilde is structurally absent for OBC, so it is not mandatory.
                od = od[np.isfinite(od[:, :3]).all(axis=1)]
                data[L] = (gd, od, src)
                print(f"  OK   L={L:2d}  ({src:4s})  gap:{gd.shape}  obs:{od.shape}")
            except Exception as e:
                print(f"  FAIL L={L}: {e}")
        else:
            for p in [gf, of]:
                tag = "OK" if p.exists() else "MISSING"
                print(f"  {tag:7s} {p.name}")
    return data


def _read_chiz_fd_header(path: Path) -> dict[str, str]:
    meta: dict[str, str] = {}
    try:
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                if not line.startswith("#"):
                    continue
                body = line[1:].strip()
                if "=" not in body:
                    continue
                key, value = body.split("=", 1)
                meta[key.strip()] = value.strip()
    except OSError as exc:
        print(f"  WARN {path.name}: cannot read header: {exc}")
    return meta


def load_chiz_fd_data(
    dh_tag: str = "dh_5e-04",
    sizes=ALL_SIZES,
    pbc: bool = True,
    allow_partial: bool = False,
) -> dict:
    """Load optional finite-difference chi_z data for plotting only."""
    data: dict = {}
    bc_label = "PBC" if pbc else "OBC"
    base_dir = CHIZ_FD_DIR / dh_tag / bc_label

    for L in sizes:
        fname = f"chizfd_L{L:02d}.dat" if pbc else f"chizfd_obc_L{L:02d}.dat"
        path = base_dir / fname
        if not path.exists():
            print(f"  SKIP chiz_fd {bc_label} L={L:02d}: file not found")
            continue

        meta = _read_chiz_fd_header(path)
        try:
            n_expected = int(meta["n_g"]) if "n_g" in meta else None
        except ValueError:
            print(f"  WARN chiz_fd {bc_label} L={L:02d}: invalid n_g={meta.get('n_g')!r}")
            n_expected = None

        try:
            arr = np.genfromtxt(path, comments="#")
        except Exception as exc:
            print(f"  WARN chiz_fd {bc_label} L={L:02d}: cannot load {path.name}: {exc}")
            continue
        arr = np.atleast_2d(arr)
        if arr.size == 0 or arr.shape[1] != 10:
            nf = arr.shape[1] if arr.ndim == 2 else 0
            print(f"  WARN chiz_fd {bc_label} L={L:02d}: NF={nf} (expected 10), skipped")
            continue

        finite_rows = np.isfinite(arr).all(axis=1)
        if not np.all(finite_rows):
            print(f"  WARN chiz_fd {bc_label} L={L:02d}: dropping {np.count_nonzero(~finite_rows)} non-finite rows")
            arr = arr[finite_rows]
        if arr.shape[0] == 0:
            print(f"  WARN chiz_fd {bc_label} L={L:02d}: no finite rows after filtering")
            continue

        n_rows = int(arr.shape[0])
        n_expected_eff = int(n_expected) if n_expected is not None else n_rows
        complete = n_rows == n_expected_eff
        if not complete and not allow_partial:
            print(
                f"  WARN chiz_fd {bc_label} L={L:02d}: partial {n_rows}/{n_expected_eff}, "
                "skipped"
            )
            continue

        method_vals = np.unique(arr[:, CHIZFD_COL["method_code"]].astype(int))
        method_code = int(method_vals[0]) if method_vals.size else -1
        if method_vals.size > 1:
            print(f"  WARN chiz_fd {bc_label} L={L:02d}: multiple method_code values {method_vals}")
        expected_method = 0 if int(L) <= ED_SIZES[-1] else 1
        if method_code != expected_method:
            print(
                f"  WARN chiz_fd {bc_label} L={L:02d}: method_code={method_code}, "
                f"expected {expected_method}"
            )

        g_arr = arr[:, CHIZFD_COL["g"]]
        chi_fd = arr[:, CHIZFD_COL["chi_fd"]]
        plot_mask = np.isfinite(g_arr) & np.isfinite(chi_fd) & (chi_fd > 0.0)
        if np.count_nonzero(plot_mask) < 1:
            print(f"  WARN chiz_fd {bc_label} L={L:02d}: no positive finite chi_fd rows")
            continue

        data[int(L)] = {
            "g": g_arr[plot_mask],
            "chi": chi_fd[plot_mask],
            "method": method_code,
            "complete": complete,
            "n_expected": n_expected_eff,
            "n_rows": n_rows,
            "path": path,
        }
        method_label = "ED" if method_code == 0 else "LNCZ" if method_code == 1 else f"code={method_code}"
        status = "complete" if complete else "partial"
        print(
            f"  OK   chiz_fd {bc_label} L={L:2d} ({method_label:4s}) "
            f"rows={n_rows}/{n_expected_eff} {status}"
        )

    return data


def save_fig(fig: plt.Figure, name: str) -> None:
    path = PLOT_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path.name}")


def read_fss_results(path: Path) -> dict:
    """Return measured exponents from the machine-readable section of fss_results.txt."""
    out: dict[str, float] = {}
    if not path.exists():
        return out
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return out

    in_kv = False
    for line in text.splitlines():
        stripped = line.strip()
        if "MACHINE-READABLE KEY-VALUE" in stripped:
            in_kv = True
            continue
        if not in_kv:
            continue
        if stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, val = stripped.partition("=")
        try:
            out[key.strip()] = float(val.strip())
        except ValueError:
            pass

    return out


def apply_grid(ax: plt.Axes) -> None:
    """Apply the shared internal plot grid style."""
    ax.grid(True, which="major", axis="both", color="0.70",
            alpha=0.55, linestyle=":", linewidth=0.8, zorder=0)


class _DoubleBCLine:
    def __init__(self, L: int, linestyle: str = "-") -> None:
        self.L = int(L)
        self.linestyle = linestyle


class _DoubleBCLineHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        y_pbc = y0 + 0.68 * height
        y_obc = y0 + 0.32 * height
        artists = [
            Line2D(
                [x0, x0 + width], [y_pbc, y_pbc],
                color=COLORS_PBC.get(orig_handle.L, "gray"),
                ls=orig_handle.linestyle,
                lw=1.9,
                solid_capstyle="round",
            ),
            Line2D(
                [x0, x0 + width], [y_obc, y_obc],
                color=COLORS_OBC.get(orig_handle.L, "gray"),
                ls=orig_handle.linestyle,
                lw=1.9,
                solid_capstyle="round",
            ),
        ]
        for artist in artists:
            artist.set_transform(handlebox.get_transform())
            handlebox.add_artist(artist)
        return artists[0]


def _method_code_from_source(source, L: int) -> int:
    if isinstance(source, str):
        source_upper = source.upper()
        if source_upper == "ED":
            return 0
        if source_upper in {"LNCZ", "LANCZOS", "LZ"}:
            return 1
    try:
        method = int(source)
        if method in (0, 1):
            return method
    except (TypeError, ValueError):
        pass
    return 0 if int(L) <= ED_SIZES[-1] else 1


def _register_external_legend_item(
    plotted_L_methods: dict[int, set[int]],
    plotted_methods: set[int],
    L: int,
    method,
) -> None:
    method_code = _method_code_from_source(method, int(L))
    plotted_L_methods.setdefault(int(L), set()).add(method_code)
    if method_code in (0, 1):
        plotted_methods.add(method_code)


def _linestyle_for_method_set(methods: set[int]) -> str:
    if 1 in methods:
        return "--"
    return "-"


def _external_size_method_legend(
    fig: plt.Figure,
    plotted_L_methods: dict[int, set[int]],
    plotted_methods: set[int],
    *,
    encode_method_in_size: bool = True,
    y_anchor: float = EXTERNAL_LEGEND_Y_ANCHOR,
) -> None:
    handles = []
    labels = []
    for L in ALL_SIZES:
        if L not in plotted_L_methods:
            continue
        linestyle = _linestyle_for_method_set(plotted_L_methods[L]) if encode_method_in_size else "-"
        handles.append(_DoubleBCLine(L, linestyle))
        labels.append(rf"$L={L}$")

    if 0 in plotted_methods:
        handles.append(Line2D([], [], color="gray", ls="-", lw=1.9))
        labels.append("ED")
    if 1 in plotted_methods:
        handles.append(Line2D([], [], color="gray", ls="--", lw=1.9))
        labels.append("LNCZ")

    if not handles:
        return

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, y_anchor),
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        ncol=6,
        columnspacing=1.7,
        handlelength=2.2,
        handletextpad=0.8,
        handler_map={_DoubleBCLine: _DoubleBCLineHandler()},
    )


def add_legend(ax: plt.Axes, data: dict, *, loc: str | None = None) -> None:
    """
    Build a legend with:
      - one entry per L (coloured, with appropriate linestyle)
      - two generic entries: solid='ED', dashed='LNCZ'
    """
    from matplotlib.lines import Line2D

    handles, labels = ax.get_legend_handles_labels()

    # Add compact entries for the numerical backend.
    has_ed = any(src == "ED" for (_, _, src) in data.values())
    has_lz = any(src == "LNCZ" for (_, _, src) in data.values())
    if has_ed:
        handles.append(Line2D([], [], color="black", ls="-", lw=1.8))
        labels.append("ED")
    if has_lz:
        handles.append(Line2D([], [], color="black", ls="--", lw=1.8))
        labels.append("LNCZ")

    legend_kwargs = {}
    if loc is not None:
        legend_kwargs["loc"] = loc
    ax.legend(handles, labels, frameon=False, fontsize=LEGEND_FONTSIZE, ncol=2,
              **legend_kwargs)


# ---------------------------------------------------------------------------
# Longitudinal sweep of transverse magnetization
# ---------------------------------------------------------------------------
def plot_mx(data_pbc: dict, data_obc: dict | None = None) -> None:
    data_obc = data_obc or {}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True,
                             sharex=True, sharey=True)
    plotted_L_methods: dict[int, set[int]] = {}
    plotted_methods: set[int] = set()
    g_th = np.linspace(0.5, 1.5, 600)
    nq = 8192
    q = (np.arange(nq, dtype=float) + 0.5) * np.pi / nq
    cos_q = np.cos(q)
    x = 1.0 / g_th[:, None]
    omega = np.sqrt(1.0 + x * x + 2.0 * x * cos_q[None, :])
    inv_omega = 1.0 / omega
    mx_th = np.mean(inv_omega, axis=1) + (1.0 / g_th) * np.mean(cos_q[None, :] * inv_omega, axis=1)

    def add_mx_info_legend(
        ax: plt.Axes,
        *,
        info_loc: str,
        theory_color: str,
    ) -> None:
        info_handles = [
            Line2D([], [], color="gray", lw=1.5, ls=":",
                   label=r"$g_C=1$"),
            Line2D([], [], color=theory_color, lw=1.5, ls="-",
                   label=r"$m_x^{(\infty)}$"),
        ]
        ax.legend(handles=info_handles, frameon=False, fontsize=LEGEND_FONTSIZE,
                  loc=info_loc)

    for ax, dset, colors, bc_label, size_loc, info_loc in [
        (axes[0], data_pbc, COLORS_PBC, "PBC", "upper left", "lower right"),
        (axes[1], data_obc, COLORS_OBC, "OBC", "lower right", "upper left"),
    ]:
        theory_color = MX_THEORY_COLORS[bc_label]
        ax.axvline(1.0, color="gray", ls=":", lw=1.5, alpha=0.5, zorder=1)
        for L in sorted(dset.keys()):
            gd, od, src = dset[L]
            ls = "-" if src == "ED" else "--"
            ax.plot(od[:, COL["g"]], od[:, COL["Mx"]], color=colors[L],
                    ls=ls, lw=1.5, zorder=2)
            _register_external_legend_item(plotted_L_methods, plotted_methods, int(L), src)
        ax.plot(g_th, mx_th, color=theory_color, ls="-", lw=1.5,
                zorder=4)
        ax.set_xlim(0.5, 1.5)
        ax.set_xlabel(r"$g$", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(r"$m_x$", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_title(bc_label, loc="right", fontsize=TITLE_FONTSIZE)
        ax.tick_params(labelsize=TICK_FONTSIZE, direction="in", which="both",
                   top=True, right=True, labelbottom=True)
        apply_grid(ax)
        add_mx_info_legend(ax, info_loc=info_loc, theory_color=theory_color)
    axes[1].set_ylabel("")
    _external_size_method_legend(fig, plotted_L_methods, plotted_methods)
    save_fig(fig, "mx_vs_g.pdf")


# ---------------------------------------------------------------------------
# Low-energy gaps versus transverse field
# ---------------------------------------------------------------------------
def plot_delta_vs_g(data_pbc: dict, data_obc: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True,
                             sharex=True)
    plotted_L_methods: dict[int, set[int]] = {}
    plotted_methods: set[int] = set()

    delta_labels = {
        "Delta0": r"$\Delta_0$",
        "Delta1": r"$\Delta_1$",
    }
    panel_specs = [
        (axes[0, 0], data_pbc, COLORS_PBC, "Delta0", "PBC"),
        (axes[1, 0], data_pbc, COLORS_PBC, "Delta1", "PBC"),
        (axes[0, 1], data_obc, COLORS_OBC, "Delta0", "OBC"),
        (axes[1, 1], data_obc, COLORS_OBC, "Delta1", "OBC"),
    ]

    for ax, dset, colors, delta_key, bc_label in panel_specs:
        ax.axvline(1.0, color="gray", ls=":", lw=1.5, alpha=0.5, zorder=1)
        for L in sorted(dset.keys()):
            gd, od, src = dset[L]
            if gd.shape[1] < 4:
                continue
            gcols = gap_columns(gd)
            ls = "-" if src == "ED" else "--"
            ax.plot(gcols["g_arr"], gcols[delta_key],
                    color=colors.get(L, "gray"), ls=ls, lw=1.4, zorder=2)
            _register_external_legend_item(plotted_L_methods, plotted_methods, int(L), src)
        ax.set_xlim(0.5, 1.5)
        ax.set_xlabel(r"$g$", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(delta_labels.get(delta_key, delta_key),
                      fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_title(bc_label, fontsize=TITLE_FONTSIZE, loc="right")
        ax.tick_params(labelsize=TICK_FONTSIZE, direction="in", which="both",
                   top=True, right=True, labelbottom=True)
        apply_grid(ax)
    _external_size_method_legend(fig, plotted_L_methods, plotted_methods, y_anchor=-0.08)
    save_fig(fig, "delta_vs_g.pdf")


# ---------------------------------------------------------------------------
# Two-gap spectrum at fixed size
# ---------------------------------------------------------------------------
def plot_gap_spectrum_panels(data_pbc: dict, data_obc: dict,
                             target_L: int = 20) -> None:
    """
    Two-panel gap spectrum at fixed L:
            left:  PBC (Delta_0 and Delta_1)
            right: OBC (same L, Delta_0 and Delta_1)
      plus thermodynamic asymptotes: PBC uses 4|g-1| for g<1
      and 2|g-1| for g>1; OBC uses 2|g-1|.
    """
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True,
                             sharex=True)
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.03, wspace=0.01, hspace=0.02)
    plotted_L_methods: dict[int, set[int]] = {}
    plotted_methods: set[int] = set()

    common_L = sorted(set(data_pbc.keys()) & set(data_obc.keys()))
    candidates = [L for L in common_L if L <= target_L]
    L_plot = max(candidates) if candidates else None

    if L_plot is None:
        for ax in axes:
            ax.set_title("No common L available for PBC/OBC", fontsize=TITLE_FONTSIZE)
        save_fig(fig, "gap_spectrum_panels.pdf")
        return

    gd_pbc, _, _ = data_pbc[L_plot]
    gd_obc, _, _ = data_obc[L_plot]

    pbc_cols = gap_columns(gd_pbc)
    obc_cols = gap_columns(gd_obc)

    g_pbc = pbc_cols["g_arr"]
    g_obc = obc_cols["g_arr"]
    d0_pbc = pbc_cols["Delta0"]
    d0_obc = obc_cols["Delta0"]
    d1_pbc = pbc_cols["Delta1"]
    d1_obc = obc_cols["Delta1"]

    all_g = np.concatenate([g_pbc, g_obc])
    g_min = float(np.nanmin(all_g))
    g_max = float(np.nanmax(all_g))
    if g_max <= g_min:
        g_min -= 1.0
        g_max += 1.0
    g_norm = plt.Normalize(vmin=g_min, vmax=g_max)
    cmap_pbc = plt.cm.plasma
    cmap_obc = plt.cm.viridis

    c_pbc = cmap_pbc(g_norm(g_pbc))
    c_obc = cmap_obc(g_norm(g_obc))

    g_line = np.linspace(0.4, 1.6, 600)
    asym_pbc = np.where(g_line < 1.0,
                        4.0 * np.abs(g_line - 1.0),
                        2.0 * np.abs(g_line - 1.0))
    asym_obc = 2.0 * np.abs(g_line - 1.0)

    # Left panel: PBC only (same L as OBC panel)
    ax = axes[0]
    ax.plot(g_line, asym_pbc, color="steelblue", lw=1.5, ls="-", zorder=1)
    ax.scatter(g_pbc, d0_pbc, marker="o", s=26, c=g_pbc, cmap=cmap_pbc,
        vmin=g_min, vmax=g_max, linewidths=0.0, zorder=3)
    ax.scatter(g_pbc, d1_pbc, marker="o", s=26, facecolors="none",
        edgecolors=c_pbc, linewidths=1.2, zorder=3)
    ax.axvline(1.0, color="gray", ls=":", lw=1.2, alpha=0.5)
    ax.set_ylabel(r"$\Delta$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylim(0.0, 4.0)
    ax.set_title(rf"PBC, $L={L_plot}$", fontsize=TITLE_FONTSIZE, loc="right")
    ax.tick_params(labelsize=TICK_FONTSIZE, direction="in", which="both", top=True, right=True)
    apply_grid(ax)
    ax.legend(handles=[
        Line2D([], [], color="black", lw=0, marker="o", markersize=5,
               markerfacecolor="black", label=r"$\Delta_0$"),
        Line2D([], [], color="black", lw=0, marker="o", markersize=5,
               markerfacecolor="none", markeredgewidth=1.2,
               label=r"$\Delta_1$"),
    ], frameon=False, fontsize=LEGEND_FONTSIZE)

    # Right panel: OBC only (same L as PBC panel)
    ax = axes[1]
    ax.plot(g_line, asym_obc, color="steelblue", lw=1.5, ls="-", zorder=1)
    ax.scatter(g_obc, d0_obc, marker="^", s=30, c=g_obc, cmap=cmap_obc,
        vmin=g_min, vmax=g_max, linewidths=0.0, zorder=3)
    ax.scatter(g_obc, d1_obc, marker="^", s=30, facecolors="none",
        edgecolors=c_obc, linewidths=1.2, zorder=3)
    ax.axvline(1.0, color="gray", ls=":", lw=1.2, alpha=0.5)
    ax.set_ylim(0.0, 2.0)
    ax.set_title(rf"OBC, $L={L_plot}$", fontsize=TITLE_FONTSIZE, loc="right")
    ax.tick_params(labelsize=TICK_FONTSIZE, direction="in", which="both", top=True, right=True)
    apply_grid(ax)
    ax.legend(handles=[
        Line2D([], [], color="black", lw=0, marker="^", markersize=5,
               markerfacecolor="black", label=r"$\Delta_0$"),
        Line2D([], [], color="black", lw=0, marker="^", markersize=5,
               markerfacecolor="none", markeredgewidth=1.2,
               label=r"$\Delta_1$"),
    ], frameon=False, fontsize=LEGEND_FONTSIZE)

    for ax in axes:
        ax.set_xlabel(r"$g$", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_xlim(0.4, 1.5)
        ax.set_xticks(np.arange(0.6, 1.41, 0.2))
    save_fig(fig, "gap_spectrum_panels.pdf")


def plot_chiz_fd_raw_panels(chiz_pbc: dict, chiz_obc: dict) -> None:
    """Raw official finite-difference chi_z from data/h_null/chiz_fd, without FSS fits."""
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True,
                             sharex=True)
    plotted_L_methods: dict[int, set[int]] = {}
    plotted_methods: set[int] = set()

    for ax, bc_label, dataset, colors in [
        (axes[0], "PBC", chiz_pbc, COLORS_PBC),
        (axes[1], "OBC", chiz_obc, COLORS_OBC),
    ]:
        has_data = False
        has_ed = False
        has_lz = False

        for L in sorted(dataset.keys()):
            item = dataset[L]
            g_arr = np.asarray(item["g"], dtype=float)
            chi = np.asarray(item["chi"], dtype=float)
            m = np.isfinite(g_arr) & np.isfinite(chi) & (chi > 0.0)
            if np.count_nonzero(m) < 1:
                continue

            method = int(item.get("method", 0 if int(L) <= ED_SIZES[-1] else 1))
            ls = "-" if method == 0 else "--"
            has_ed = has_ed or method == 0
            has_lz = has_lz or method == 1
            label = rf"$L={L}$"
            if not bool(item.get("complete", True)):
                label = rf"$L={L}$ partial"
            ax.plot(g_arr[m], chi[m], color=colors.get(L, "gray"), ls=ls,
                    lw=1.35, label=label, zorder=2)
            _register_external_legend_item(plotted_L_methods, plotted_methods, int(L), method)
            has_data = True

        ax.axvline(1.0, color="gray", ls=":", lw=1.2, alpha=0.6, zorder=1)
        ax.set_xlim(0.4, 1.6)
        ax.set_yscale("log")
        ax.set_xlabel(r"$g$", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_title(bc_label, loc="right", fontsize=TITLE_FONTSIZE)
        ax.tick_params(direction="in", which="both", top=True, right=True,
                       labelsize=TICK_FONTSIZE)
        apply_grid(ax)

        if not has_data:
            ax.text(0.5, 0.5, "No finite FD data",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=TEXT_FONTSIZE, color="gray")
            continue

    axes[0].set_ylabel(r"$\chi_z$", fontsize=AXIS_LABEL_FONTSIZE)
    axes[1].set_ylabel("")
    axes[1].tick_params(labelleft=False)
    _external_size_method_legend(fig, plotted_L_methods, plotted_methods)
    save_fig(fig, "chi_z.pdf")

def plot_psi_panels(data_pbc: dict, data_obc: dict) -> None:
    """
    Plot psi_tilde and psi_bar versus g for both boundary conditions.
    Top row: psi_tilde(g) per L; this channel is absent when symmetry sectors are unavailable.
    Bottom row: psi_bar(g) per L.
    Line style: ED solid, Lanczos dashed. Colors: PBC plasma, OBC viridis.
    """
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True,
                             sharex=True, sharey=True)
    plotted_L_methods: dict[int, set[int]] = {}
    plotted_methods: set[int] = set()

    panel_specs = [
        ("PBC", data_pbc, COLORS_PBC, 0),
        ("OBC", data_obc, COLORS_OBC, 1),
    ]
    row_specs = [
        (COL["psi_tilde"], r"$\tilde{\Psi}(L,g)$"),
        (COL["psi_bar"], r"$\bar{\Psi}(L,g)$"),
    ]

    for bc_label, data, colors, col in panel_specs:
        for row, (obs_col, ylabel) in enumerate(row_specs):
            ax = axes[row, col]
            has_data = False

            for L in sorted(data.keys()):
                gd, od, src = data[L]
                g = od[:, COL["g"]]
                y = od[:, obs_col]
                mask = np.isfinite(g) & np.isfinite(y)
                if not mask.any():
                    continue

                ls = "-" if src == "ED" else "--"
                ax.plot(g[mask], y[mask],
                        color=colors.get(L, "gray"), ls=ls, lw=1.5,
                        label=rf"$L={L}$", zorder=2)
                _register_external_legend_item(plotted_L_methods, plotted_methods, int(L), src)
                has_data = True

            ax.axvline(1.0, color="gray", ls=":", lw=1.5, alpha=0.5, zorder=1)
            gc_legend = ax.legend(
                handles=[
                    Line2D([], [], color="gray", ls=":", lw=1.5,
                           label=r"$g_c=1$")
                ],
                frameon=False, fontsize=LEGEND_FONTSIZE, loc="upper right",
            )
            ax.add_artist(gc_legend)
            ax.set_xlim(0.4, 1.5)
            ax.set_xticks(np.arange(0.6, 1.41, 0.2))
            ax.set_ylim(0.0, 1.05)
            ax.tick_params(labelsize=TICK_FONTSIZE, direction="in", which="both",
                           top=True, right=True)
            apply_grid(ax)

            ax.set_title(bc_label, loc="right", fontsize=TITLE_FONTSIZE)
            ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
            ax.set_xlabel(r"$g$", fontsize=AXIS_LABEL_FONTSIZE)
            ax.tick_params(labelbottom=True, labelleft=True)

            if not has_data:
                ax.text(0.5, 0.5, "No finite data",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=TEXT_FONTSIZE, color="gray")

    _external_size_method_legend(fig, plotted_L_methods, plotted_methods, y_anchor=-0.08)
    save_fig(fig, "psi_panels.pdf")

# ---------------------------------------------------------------------------
# Transverse susceptibility panels
# ---------------------------------------------------------------------------
def plot_transverse_chi_panels(data_pbc: dict, data_obc: dict) -> None:
    """Two panels: PBC and OBC g·chi_x(g) curves."""
    fss_vals = read_fss_results(FSS_DIR / "fss_results.txt")
    g_pc_by_bc = {
        "PBC": fss_vals.get("GPC_PBC_SUBLEADING", 1.0),
        "OBC": fss_vals.get("GPC_OBC_MIXED_SUBLEADING", 1.0),
    }
    pbc_color = plt.cm.plasma(0.62)
    obc_color = plt.cm.viridis(0.62)

    fig, (ax_pbc_top, ax_obc_top) = plt.subplots(
        1, 2, figsize=(14, 5.5), constrained_layout=True, sharey=True
    )
    plotted_L_methods: dict[int, set[int]] = {}
    plotted_methods: set[int] = set()

    def plot_bc(ax_top: plt.Axes, dset: dict, bc_label: str, base_color) -> None:
        L_list = sorted(dset.keys())
        colors = colors_for_bc(bc_label)
        g_pc = float(g_pc_by_bc.get(bc_label, 1.0))
        if not np.isfinite(g_pc):
            g_pc = 1.0

        for L in L_list:
            gd, od, src = dset[L]
            g = od[:, COL["g"]]
            if od.shape[1] <= COL["g_chi_perp"]:
                continue
            chi_p = od[:, COL["chi_perp"]]
            g_chi_p = od[:, COL["g_chi_perp"]]
            m = np.isfinite(g) & np.isfinite(chi_p) & np.isfinite(g_chi_p)
            if not m.any():
                continue
            ls = "-" if src == "ED" else "--"
            color = colors.get(L, base_color)
            ax_top.plot(g[m], g_chi_p[m], color=color,
                        ls=ls, lw=1.3, label=rf"$L={L}$", zorder=2)
            _register_external_legend_item(plotted_L_methods, plotted_methods, int(L), src)
        ax_top.set_title(bc_label, loc="right", fontsize=TITLE_FONTSIZE)

    plot_bc(ax_pbc_top, data_pbc, "PBC", pbc_color)
    plot_bc(ax_obc_top, data_obc, "OBC", obc_color)

    ax_pbc_top.set_xlabel(r"$g$", fontsize=AXIS_LABEL_FONTSIZE)
    ax_obc_top.set_xlabel(r"$g$", fontsize=AXIS_LABEL_FONTSIZE)
    ax_pbc_top.set_ylabel(r"$g\,\chi_x$", fontsize=AXIS_LABEL_FONTSIZE)
    ax_obc_top.set_ylabel(r"$g\,\chi_x$", fontsize=AXIS_LABEL_FONTSIZE)
    for ax in (ax_pbc_top, ax_obc_top):
        ax.set_xlim(0.4, 1.6)
        ax.axvline(1.0, color="gray", ls=":", lw=1.2, alpha=0.6)

    for ax in (ax_pbc_top, ax_obc_top):
        ax.tick_params(direction="in", which="both", top=True, right=True, labelsize=TICK_FONTSIZE)
        apply_grid(ax)

    _external_size_method_legend(fig, plotted_L_methods, plotted_methods)
    save_fig(fig, "transverse_chi_panels.pdf")


# ---------------------------------------------------------------------------
# Binder diagnostic panels
# ---------------------------------------------------------------------------
def _find_binder_crossing_local(obs_L1, obs_L2):
    """Return g_cross where U(L1, g) = U(L2, g), or NaN if not found."""
    g1, od1 = obs_L1
    g2, od2 = obs_L2
    binder_col = COL["binder"]
    if od1.shape[1] <= binder_col or od2.shape[1] <= binder_col:
        return np.nan

    g_lo = max(float(g1.min()), float(g2.min()))
    g_hi = min(float(g1.max()), float(g2.max()))
    if g_lo >= g_hi:
        return np.nan

    m1 = np.isfinite(g1) & np.isfinite(od1[:, binder_col])
    m2 = np.isfinite(g2) & np.isfinite(od2[:, binder_col])
    if m1.sum() < 4 or m2.sum() < 4:
        return np.nan

    try:
        spl1 = CubicSpline(g1[m1], od1[m1, binder_col])
        spl2 = CubicSpline(g2[m2], od2[m2, binder_col])
    except Exception:
        return np.nan

    g_grid = np.linspace(g_lo, g_hi, 1000)
    diff = spl1(g_grid) - spl2(g_grid)
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    if sign_changes.size == 0:
        return np.nan

    # Pick crossing closest to g=1
    best_gc = np.nan
    best_dist = np.inf
    for idx in sign_changes:
        try:
            gc_loc = brentq(lambda g: float(spl1(g) - spl2(g)),
                            float(g_grid[idx]), float(g_grid[idx + 1]))
            if abs(gc_loc - 1.0) < best_dist:
                best_dist = abs(gc_loc - 1.0)
                best_gc = gc_loc
        except Exception:
            pass
    return best_gc


def _binder_crossing_series(
    data: dict, g_c_num: float, bc_label: str = ""
) -> tuple[np.ndarray, np.ndarray]:
    """Return adjacent-size Binder crossings as L2 and gpc-g_cross."""
    obs = {L: (od[:, COL["g"]], od) for L, (gd, od, src) in data.items()}
    L_vals, gc_drift = [], []
    Ls = sorted(obs)
    for L1, L2 in zip(Ls[:-1], Ls[1:]):
        gc = _find_binder_crossing_local(obs[L1], obs[L2])
        if not np.isfinite(gc):
            print(
                f"  [WARN] Binder shift {bc_label}: no finite crossing for "
                f"L1={L1} L2={L2}"
            )
            continue
        y = g_c_num - gc
        if np.isfinite(y) and y > 0.0:
            L_vals.append(float(L2))
            gc_drift.append(float(y))
        else:
            print(
                f"  [WARN] Binder shift {bc_label}: discarded non-positive "
                f"g_pc-g_cross for L1={L1} L2={L2}; "
                f"g_cross={gc:.12g}, g_pc={g_c_num:.12g}, shift={y:.12g}"
            )
    return np.asarray(L_vals, dtype=float), np.asarray(gc_drift, dtype=float)


def plot_gcross_vs_L(data_pbc: dict, data_obc: dict) -> None:
    """
    Binder crossing drift: gpc-g_cross versus L on log-log axes.

    The fitted p_eff is only an effective finite-size drift exponent for the
    Binder crossing convergence. It is diagnostic and is not used in the final
    critical-exponent estimates.
    """
    LMIN_FIT = 10.0
    fss_vals = read_fss_results(FSS_DIR / "fss_results.txt")
    g_c_num = fss_vals.get("gc_num", 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True,
                             sharey=True)

    for ax, data, bc_label, color in [
        (axes[0], data_pbc, "PBC", PBC_ACCENT),
        (axes[1], data_obc, "OBC", OBC_ACCENT),
    ]:
        L_arr, y_arr = _binder_crossing_series(data, g_c_num, bc_label)
        if L_arr.size:
            ax.plot(L_arr, y_arr, marker="o", ms=6, lw=0,
                    color=color, zorder=3)

            fit_mask = (
                np.isfinite(L_arr) & np.isfinite(y_arr)
                & (L_arr >= LMIN_FIT) & (y_arr > 0.0)
            )
            L_fit_data = L_arr[fit_mask]
            y_fit_data = y_arr[fit_mask]
            if L_fit_data.size >= 3:
                logL = np.log(L_fit_data)
                logy = np.log(y_fit_data)
                slope, intercept = np.polyfit(logL, logy, 1)
                p_eff = -float(slope)
                amp = float(np.exp(intercept))

                L_line = np.linspace(float(L_fit_data.min()),
                                     float(L_fit_data.max()), 200)
                y_line = amp * L_line ** (-p_eff)
                ax.plot(L_line, y_line, ls="--", lw=1.5, color=color,
                        label=r"fit: $A L^{-p_{\rm eff}}$", zorder=2)
                ax.annotate(
                    rf"$p_{{\rm eff}} = {p_eff:.3f}$" "\n"
                    rf"$L_{{\min}} = {LMIN_FIT:.0f}$",
                    xy=(0.06, 0.08), xycoords="axes fraction",
                    fontsize=TEXT_FONTSIZE, color="black",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              alpha=0.78, ec="none"),
                )
                print(
                    f"[Binder shift] {bc_label}: p_eff={p_eff:.6f}, "
                    f"A={amp:.6e}, Lmin={LMIN_FIT:.1f}, "
                    f"n={L_fit_data.size}"
                )
            else:
                msg = "Not enough positive crossings for fit"
                ax.text(0.5, 0.5, msg, transform=ax.transAxes,
                        ha="center", va="center", fontsize=TEXT_FONTSIZE, color="gray")
                print(
                    f"  [WARN] Binder shift {bc_label}: {msg} "
                    f"(Lmin={LMIN_FIT:.1f}, n={L_fit_data.size})"
                )
        else:
            ax.text(0.5, 0.5, "Not enough positive crossings for fit",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=TEXT_FONTSIZE, color="gray")
            print(
                f"  [WARN] Binder shift {bc_label}: "
                "Not enough positive crossings for fit (n=0)"
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$L$", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_title(bc_label, loc="right", fontsize=TITLE_FONTSIZE)
        ax.tick_params(direction="in", which="both", top=True, right=True,
                       labelsize=TICK_FONTSIZE)
        apply_grid(ax)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, frameon=False, fontsize=LEGEND_FONTSIZE, loc="best")

    axes[0].set_ylabel(r"$g_{pc} - g_{\rm cross}$", fontsize=AXIS_LABEL_FONTSIZE)
    save_fig(fig, "binder_crossing_shift.pdf")


def plot_binder_panels(data_pbc: dict, data_obc: dict) -> None:
    """Binder diagnostics:
      binder_all.pdf:     U vs g, split by boundary condition, with zoom insets.
      binder_summary.pdf: Binder crossing estimates plus crossing-shift panels.
    """
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter, ScalarFormatter
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    fss_vals = read_fss_results(FSS_DIR / "fss_results.txt")
    g_c_fallback = fss_vals.get("gc_num", 1.0)

    def _augment_pbc_binder_data(dset: dict) -> dict:
        out = dict(dset)
        for L in CHIZ_PBC_EXTRA_SIZES:
            if L in out:
                continue
            candidates = [
                data_dir_for_bc(True) / f"obs_lz_L{L:02d}.dat",
                data_dir_for_bc(True) / f"obs_L{L:02d}.dat",
            ]
            for path in candidates:
                if not path.exists():
                    continue
                try:
                    od = np.atleast_2d(np.loadtxt(path, comments="#"))
                except Exception as exc:
                    print(f"  WARN binder extra PBC L={L:02d}: {exc}")
                    continue
                if od.size == 0 or od.shape[1] < 3:
                    continue
                od = od[np.isfinite(od[:, :3]).all(axis=1)]
                if od.size == 0:
                    continue
                src = "LNCZ" if "lz" in path.name else "ED"
                out[L] = (None, od, src)
                print(f"  OK   Binder extra PBC L={L:2d} ({src:4s}) obs:{od.shape}")
                break
        return out

    data_pbc_use = _augment_pbc_binder_data(data_pbc)
    data_obc_use = data_obc

    fig_top, axes_top = plt.subplots(
        1, 2, figsize=(14, 5.5), constrained_layout=True,
    )
    plotted_L_methods: dict[int, set[int]] = {}
    plotted_methods: set[int] = set()
    ax_u_pbc, ax_u_obc = axes_top
    ax_zoom_pbc = inset_axes(
        ax_u_pbc, width="45%", height="45%", loc="upper left",
        bbox_to_anchor=(0.06, 0.0, 1.0, 1.0),
        bbox_transform=ax_u_pbc.transAxes, borderpad=1.0,
    )
    ax_zoom_obc = inset_axes(
        ax_u_obc, width="45%", height="45%", loc="upper left",
        bbox_to_anchor=(0.05, 0.0, 1.0, 1.0),
        bbox_transform=ax_u_obc.transAxes, borderpad=1.0,
    )
    for zoom_ax in (ax_zoom_pbc, ax_zoom_obc):
        zoom_ax.patch.set_facecolor("white")
        zoom_ax.patch.set_alpha(0.88)

    def add_binder_legend(ax: plt.Axes, dset: dict, colors: dict) -> None:
        handles = [
            Line2D([], [], color=colors.get(L, "gray"), lw=1.4,
                   ls="-" if dset[L][2] == "ED" else "--",
                   label=rf"$L={L}$")
            for L in sorted(dset.keys())
        ]
        sources = [src for (_, _, src) in dset.values()]
        if "ED" in sources:
            handles.append(Line2D([], [], color="black", ls="-", lw=1.5,
                                  label="ED"))
        if "LNCZ" in sources:
            handles.append(Line2D([], [], color="black", ls="--", lw=1.5,
                                  label="LNCZ"))
        ax.legend(handles=handles, frameon=False, fontsize=LEGEND_FONTSIZE - 1, ncol=1,
              loc="upper left")

    def plot_binder_bc(ax: plt.Axes, dset: dict, bc_label: str) -> None:
        colors = colors_for_bc(bc_label)
        for L in sorted(dset.keys()):
            gd, od, src = dset[L]
            g = od[:, COL["g"]]
            u = od[:, COL["binder"]]
            m = np.isfinite(u) & np.isfinite(g)
            if not m.any():
                continue
            ls = "-" if src == "ED" else "--"
            ax.plot(g[m], u[m], color=colors.get(L, "gray"), ls=ls, lw=1.3, zorder=2)
            _register_external_legend_item(plotted_L_methods, plotted_methods, int(L), src)

        ax.axvline(g_c_fallback, color="gray", ls=":", lw=1.2, alpha=0.5)
        ax.set_xlim(0.4, 1.5)
        ax.set_xticks(np.arange(0.6, 1.41, 0.2))
        ax.set_xlabel(r"$g$", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(r"$U_4$", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_title(bc_label, loc="right", fontsize=TITLE_FONTSIZE)

    def plot_binder_zoom(zoom_ax: plt.Axes, dset: dict, bc_label: str) -> None:
        colors = colors_for_bc(bc_label)
        for L in sorted(dset.keys()):
            gd, od, src = dset[L]
            g = od[:, COL["g"]]
            u = od[:, COL["binder"]]
            m = np.isfinite(u) & np.isfinite(g)
            if not m.any():
                continue
            mz = m & (g >= 0.94) & (g <= 1.02)
            if not mz.any():
                continue
            ls = "-" if src == "ED" else "--"
            zoom_ax.plot(g[mz], u[mz], color=colors.get(L, "gray"), ls=ls, lw=1.1)
        zoom_ax.axvline(g_c_fallback, color="gray", ls=":", lw=1.0, alpha=0.5)
        zoom_ax.set_xlim(0.94, 1.02)

    plot_binder_bc(ax_u_pbc, data_pbc_use, "PBC")
    plot_binder_bc(ax_u_obc, data_obc_use, "OBC")

    plot_binder_zoom(ax_zoom_pbc, data_pbc_use, "PBC")
    plot_binder_zoom(ax_zoom_obc, data_obc_use, "OBC")

    for zoom_ax in (ax_zoom_pbc, ax_zoom_obc):
        zoom_ax.set_xlabel("")
        zoom_ax.set_ylabel("")
        zoom_ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: f"{x:.2f}")
        )
        zoom_ax.tick_params(direction="in", which="both", top=True, right=True,
                            labelsize=TICK_FONTSIZE)
        apply_grid(zoom_ax)

    for ax in (ax_u_pbc, ax_u_obc):
        ax.tick_params(direction="in", which="both", top=True, right=True, labelsize=TICK_FONTSIZE)
        apply_grid(ax)

    ax_u_obc.set_ylabel("")
    ax_u_obc.tick_params(labelleft=False)
    ax_u_obc.set_ylim(ax_u_pbc.get_ylim())

    legend_handles = []
    legend_labels = []
    for L in ALL_SIZES:
        if L not in plotted_L_methods:
            continue
        linestyle = _linestyle_for_method_set(plotted_L_methods[L])
        legend_handles.append(_DoubleBCLine(L, linestyle))
        legend_labels.append(rf"$L={L}$")
    if 0 in plotted_methods:
        legend_handles.append(Line2D([], [], color="gray", ls="-", lw=1.9))
        legend_labels.append("ED")
    if 1 in plotted_methods:
        legend_handles.append(Line2D([], [], color="gray", ls="--", lw=1.9))
        legend_labels.append("LNCZ")
    if legend_handles:
        fig_top.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, EXTERNAL_LEGEND_Y_ANCHOR),
            bbox_transform=fig_top.transFigure,
            frameon=False,
            fontsize=LEGEND_FONTSIZE,
            ncol=6,
            columnspacing=1.7,
            handlelength=2.2,
            handletextpad=0.8,
            handler_map={_DoubleBCLine: _DoubleBCLineHandler()},
        )
    save_fig(fig_top, "binder_all.pdf")

    fig_summary, axes = plt.subplots(
        2, 2, figsize=(14, 10), constrained_layout=True,
        gridspec_kw={"height_ratios": [2.0, 2.0]},
    )
    ax_shift_pbc, ax_shift_obc = axes[0, 0], axes[0, 1]
    ax1, ax2 = axes[1, 0], axes[1, 1]

    def plot_binder_shift(ax: plt.Axes, dset: dict, bc_label: str, color: tuple) -> None:
        L_arr, y_arr = _binder_crossing_series(dset, g_c_fallback, bc_label)
        if L_arr.size:
            ax.plot(L_arr, y_arr, marker="o", ms=6, lw=0,
                    color=color, zorder=3)

            fit_mask = (
                np.isfinite(L_arr) & np.isfinite(y_arr)
                & (L_arr >= 10.0) & (y_arr > 0.0)
            )
            L_fit_data = L_arr[fit_mask]
            y_fit_data = y_arr[fit_mask]
            if L_fit_data.size >= 3:
                logL = np.log(L_fit_data)
                logy = np.log(y_fit_data)
                slope, intercept = np.polyfit(logL, logy, 1)
                p_eff = -float(slope)
                amp = float(np.exp(intercept))

                L_line = np.linspace(float(L_fit_data.min()),
                                     float(L_fit_data.max()), 200)
                y_line = amp * L_line ** (-p_eff)
                ax.plot(L_line, y_line, ls="--", lw=1.5, color=color,
                        label=r"fit: $A L^{-p_{\rm eff}}$", zorder=2)
                ax.annotate(
                    rf"$p_{{\rm eff}} = {p_eff:.3f}$" "\n"
                    rf"$L_{{\min}} = {10.0:.0f}$",
                    xy=(0.06, 0.08), xycoords="axes fraction",
                    fontsize=TEXT_FONTSIZE, color="black",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              alpha=0.78, ec="none"),
                )
                print(
                    f"[Binder shift] {bc_label}: p_eff={p_eff:.6f}, "
                    f"A={amp:.6e}, Lmin={10.0:.1f}, "
                    f"n={L_fit_data.size}"
                )
            else:
                msg = "Not enough positive crossings for fit"
                ax.text(0.5, 0.5, msg, transform=ax.transAxes,
                        ha="center", va="center", fontsize=TEXT_FONTSIZE, color="gray")
                print(
                    f"  [WARN] Binder shift {bc_label}: {msg} "
                    f"(Lmin={10.0:.1f}, n={L_fit_data.size})"
                )
        else:
            ax.text(0.5, 0.5, "Not enough positive crossings for fit",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=TEXT_FONTSIZE, color="gray")
            print(
                f"  [WARN] Binder shift {bc_label}: "
                "Not enough positive crossings for fit (n=0)"
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$L$", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_title(bc_label, loc="right", fontsize=TITLE_FONTSIZE)
        ax.tick_params(direction="in", which="both", top=True, right=True,
                       labelsize=TICK_FONTSIZE)
        apply_grid(ax)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, frameon=False, fontsize=LEGEND_FONTSIZE, loc="best")

    for ax, data, bc_label, color in [
        (ax_shift_pbc, data_pbc_use, "PBC", PBC_ACCENT),
        (ax_shift_obc, data_obc_use, "OBC", OBC_ACCENT),
    ]:
        plot_binder_shift(ax, data, bc_label, color)

    ax_shift_pbc.set_ylabel(r"$g_{pc} - g_{\rm cross}$", fontsize=AXIS_LABEL_FONTSIZE)

    # Crossing convergence panel.
    obs_pbc = {L: (od[:, COL["g"]], od) for L, (gd, od, src) in data_pbc_use.items()}
    obs_obc = {L: (od[:, COL["g"]], od) for L, (gd, od, src) in data_obc_use.items()}

    for obs_data, bc_label, color in [
        (obs_pbc, "PBC", PBC_ACCENT),
        (obs_obc, "OBC", OBC_ACCENT),
    ]:
        _EXCL = {4, 6, 8}
        Ls = sorted(L for L in obs_data.keys() if L not in _EXCL)
        Lavg_arr, gc_arr = [], []
        for i in range(len(Ls) - 1):
            gc = _find_binder_crossing_local(obs_data[Ls[i]], obs_data[Ls[i + 1]])
            if np.isfinite(gc):
                Lavg_arr.append(0.5 * (Ls[i] + Ls[i + 1]))
                gc_arr.append(gc)
        Lavg_arr = np.asarray(Lavg_arr, dtype=float)
        gc_arr = np.asarray(gc_arr, dtype=float)
        if Lavg_arr.size == 0:
            continue
        m = np.isfinite(gc_arr)
        if not m.any():
            continue
        ax1.plot(Lavg_arr[m], gc_arr[m], marker="o", markersize=5,
                 color=color, ls="None", label=bc_label, zorder=3)

    ax1.axhline(1.0, color="gray", ls=":", lw=1.3, alpha=0.8,
                label=r"$g_c=1$", zorder=1)
    ax1.set_xlabel(r"$(L_1+L_2)/2$", fontsize=AXIS_LABEL_FONTSIZE)
    ax1.set_ylabel(r"$g_{cross}$", fontsize=AXIS_LABEL_FONTSIZE)
    ax1.set_ylim(top=1.008)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(round(x))}" if x >= 1 else ""))
    ax1.xaxis.set_minor_formatter(NullFormatter())
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax1.legend(frameon=False, fontsize=LEGEND_FONTSIZE, ncol=1, loc="lower right")

    # Binder value at the pseudo-critical point.
    g_c_num = g_c_fallback  # numerical g_pc from fss_results.txt; fallback 1.0

    ax2_legend_items: dict[str, tuple[Line2D, list[Line2D]]] = {}
    for obs_data, bc_label, color in [
        (obs_pbc, "PBC", PBC_ACCENT),
        (obs_obc, "OBC", OBC_ACCENT),
    ]:
        L_vals, U_vals = [], []
        for L in sorted(obs_data.keys()):
            g_arr, od = obs_data[L]
            if od.shape[1] <= COL["binder"]:
                continue
            try:
                m_b = np.isfinite(g_arr) & np.isfinite(od[:, COL["binder"]])
                if m_b.sum() < 4:
                    continue
                spl = CubicSpline(g_arr[m_b], od[m_b, COL["binder"]])
                u_gc = float(spl(g_c_num))
                if np.isfinite(u_gc):
                    L_vals.append(float(L))
                    U_vals.append(u_gc)
            except Exception:
                pass

        invL = np.asarray([1.0 / L for L in L_vals], dtype=float)
        U_arr = np.asarray(U_vals, dtype=float)
        if invL.size < 2:
            continue

        ax2.plot(invL, U_arr, marker="o", ls="None", color=color,
                 markersize=7, label=bc_label, zorder=3)

        # Linear fit U(1/L) = U* + c/L
        coeffs = np.polyfit(invL, U_arr, 1)
        c_slope, U_star = coeffs
        invL_fit = np.linspace(0.0, invL.max() * 1.1, 200)
        ax2.plot(invL_fit, U_star + c_slope * invL_fit, ls="--", lw=1.4,
                 color=color, zorder=2)
        value_handles = [
            Line2D([], [], color="none", label=rf"$U^*={U_star:.4f}$")
        ]
        ax2_legend_items[bc_label] = (
            Line2D([], [], color=color, marker="o", ls="None", label=bc_label),
            value_handles,
        )

    ax2.set_xlabel(r"$1/L$", fontsize=AXIS_LABEL_FONTSIZE)
    ax2.set_ylabel(r"$U(g_{pc}, L)$", fontsize=AXIS_LABEL_FONTSIZE)
    legend_handles = []
    for bc_label in ("PBC", "OBC"):
        if bc_label in ax2_legend_items:
            marker_handle, value_handles = ax2_legend_items[bc_label]
            legend_handles.append(marker_handle)
            legend_handles.extend(value_handles)
    ax2.legend(handles=legend_handles, frameon=False, fontsize=LEGEND_FONTSIZE, ncol=1)
    ax2.xaxis.set_major_formatter(ScalarFormatter())
    ax2.ticklabel_format(axis="x", style="plain")

    for ax in (ax1, ax2):
        ax.tick_params(direction="in", which="both", top=True, right=True, labelsize=TICK_FONTSIZE)
        apply_grid(ax)

    save_fig(fig_summary, "binder_summary.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # Apply publication-quality style via seaborn (plasma palette set manually)
    sns.set_theme(style="ticks", context="paper", font_scale=FONT_SCALE)
    plt.rcParams.update({
        "axes.spines.right": True,
        "axes.spines.top":   True,
        "xtick.direction":   "in",
        "ytick.direction":   "in",
        "mathtext.fontset":  "cm",
        "font.family":       "serif",
        "font.size":         TEXT_FONTSIZE,
        "axes.labelsize":    AXIS_LABEL_FONTSIZE,
        "axes.titlesize":    TITLE_FONTSIZE,
        "xtick.labelsize":   TICK_FONTSIZE,
        "ytick.labelsize":   TICK_FONTSIZE,
        "legend.fontsize":   LEGEND_FONTSIZE,
    })

    print("[plot_observables.py]  Loading PBC data ...")
    data_pbc = load_data(pbc=True)

    print("[plot_observables.py]  Loading OBC data ...")
    data_obc = load_data(pbc=False)

    print("[plot_observables.py]  Loading chi_z FD PBC data ...")
    chiz_pbc = load_chiz_fd_data(pbc=True, sizes=CHIZ_PBC_SIZES)

    print("[plot_observables.py]  Loading chi_z FD OBC data ...")
    chiz_obc = load_chiz_fd_data(pbc=False, sizes=CHIZ_OBC_SIZES)

    if not data_pbc and not data_obc:
        print("[ERROR] No data found. Run ising_static / ising_lanczos first.")
        return

    output_snapshot = snapshot_outputs()

    print(f"\n[plot_observables.py]  Generating plots -> plots/h_null/observables/")
    plot_gap_spectrum_panels(data_pbc, data_obc, target_L=22)
    plot_mx(data_pbc, data_obc)
    plot_delta_vs_g(data_pbc, data_obc)
    if chiz_pbc or chiz_obc:
        plot_chiz_fd_raw_panels(chiz_pbc, chiz_obc)
    plot_psi_panels(data_pbc, data_obc)
    plot_transverse_chi_panels(data_pbc, data_obc)
    plot_binder_panels(data_pbc, data_obc)

    report_output_changes(output_snapshot)

    n_plots = len(list(PLOT_DIR.glob("*.pdf")))
    print(f"\n[plot_observables.py]  Done.  {n_plots} PDF files written.")


if __name__ == "__main__":
    main()
