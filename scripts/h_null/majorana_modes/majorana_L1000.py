#!/usr/bin/env python3
"""
Generate the L=1000 Majorana profile PDF plus matching MP4/GIF media.

Direct finite-BdG media diagonalise the finite matrix for each frame.
For L=1000 that would be unnecessarily expensive, so this script uses the
large-L OBC edge-mode envelope in the ordered phase:

    rho_j proportional to g^(2(j-1)) + g^(2(L-j)).

For g >= 1 the plotted profile is a delocalised standing-wave diagnostic,
matching the "bulk mode, no edge localization" visual role of the original
media.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    mpl_cache_dir = Path("/tmp/qising_1d_matplotlib_cache/majorana")
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache_dir)

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from majorana_modes import (
    AXIS_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    PROJECT_ROOT,
    TEXT_FONTSIZE,
    TICK_FONTSIZE,
    configure_style,
)


L_DEFAULT = 1000
OUT_DIR = PROJECT_ROOT / "plots" / "h_null" / "majorana" / "L1000"
PROFILE_G_VALUES = (0.5, 0.7, 0.85, 0.95)
LEFT_XLIM = (0.0, 35.0)
RIGHT_XLIM = (965.0, 1000.0)

FPS_VIDEO = 60
DURATION_SEC_VIDEO = 15
TOTAL_FRAMES_VIDEO = FPS_VIDEO * DURATION_SEC_VIDEO
FRAMES_TOPO_VIDEO = int(TOTAL_FRAMES_VIDEO * 0.75)
FRAMES_TRIVIAL_VIDEO = TOTAL_FRAMES_VIDEO - FRAMES_TOPO_VIDEO

FPS_GIF = 15
DURATION_SEC_GIF = 8
TOTAL_FRAMES_GIF = FPS_GIF * DURATION_SEC_GIF
FRAMES_TOPO_GIF = int(TOTAL_FRAMES_GIF * 0.70)
FRAMES_TRIVIAL_GIF = TOTAL_FRAMES_GIF - FRAMES_TOPO_GIF


def profile_density_large_l(L: int, g: float) -> np.ndarray:
    """Fast normalized profile used for the L=1000 visualizations."""
    sites0 = np.arange(L, dtype=float)
    if 0.0 < g < 1.0:
        log_g2 = 2.0 * np.log(g)
        log_left = sites0 * log_g2
        log_right = (float(L - 1) - sites0) * log_g2
        shift = max(float(np.max(log_left)), float(np.max(log_right)))
        rho = np.exp(log_left - shift) + np.exp(log_right - shift)
    else:
        sites = np.arange(1, L + 1, dtype=float)
        rho = np.sin(np.pi * sites / float(L + 1)) ** 2
    norm = float(np.sum(rho))
    if norm > 0.0:
        rho /= norm
    return rho


def timeline(topological_frames: int, trivial_frames: int) -> np.ndarray:
    """Sweep from the ordered edge-mode regime through the transition."""
    return np.concatenate([
        np.linspace(0.10, 0.99, topological_frames),
        np.linspace(0.99, 1.30, trivial_frames),
    ])


def xi_label(g: float) -> str:
    """Small on-screen label for the edge-mode decay length."""
    if 0.0 < g < 1.0:
        return rf"$\xi_M \simeq {-1.0 / np.log(g):.2f}$ sites"
    return r"bulk mode, no edge localization"


def add_break_marks(ax_left: plt.Axes, ax_right: plt.Axes, color: str) -> None:
    """Draw diagonal marks between the two pieces of a broken x axis."""
    ax_left.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    ax_right.tick_params(labelleft=False)

    d = 0.018
    kwargs = dict(color=color, clip_on=False, lw=1.1)
    ax_left.plot((1 - d, 1 + d), (-d, +d), transform=ax_left.transAxes, **kwargs)
    ax_left.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_left.transAxes, **kwargs)
    ax_right.plot((-d, +d), (-d, +d), transform=ax_right.transAxes, **kwargs)
    ax_right.plot((-d, +d), (1 - d, 1 + d), transform=ax_right.transAxes, **kwargs)


def make_broken_axes(
    *,
    figsize: tuple[float, float],
    facecolor: str | None = None,
    gridspec_kw: dict | None = None,
) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    """Create the common two-panel edge-window layout for L=1000."""
    if gridspec_kw is None:
        gridspec_kw = {"width_ratios": [1, 1], "wspace": 0.08}
    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=figsize,
        facecolor=facecolor,
        gridspec_kw=gridspec_kw,
    )
    ax_left.set_xlim(*LEFT_XLIM)
    ax_right.set_xlim(*RIGHT_XLIM)
    return fig, ax_left, ax_right


def save_pdf(L: int = L_DEFAULT) -> Path:
    """Save a report-style profile plot for the L=1000 edge windows."""
    configure_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sites = np.arange(1, L + 1)
    colors = plt.cm.Blues(np.linspace(0.9, 0.25, len(PROFILE_G_VALUES)))
    marker_step = max(1, L // 22)

    fig, ax_left, ax_right = make_broken_axes(figsize=(7.6, 4.8))
    handles = []
    for g, color in zip(PROFILE_G_VALUES, colors):
        rho = profile_density_large_l(L, float(g))
        for ax in (ax_left, ax_right):
            ax.fill_between(sites, 0.0, rho, color=color, alpha=0.22, zorder=1)
            line, = ax.plot(
                sites,
                rho,
                "-o",
                color=color,
                lw=1.5,
                ms=3.0,
                markevery=marker_step,
                label=rf"$g={g:.2f}$",
                zorder=2,
            )
            if ax is ax_left:
                handles.append(line)

    ax_left.set_ylabel(r"$\rho_j$", fontsize=AXIS_LABEL_FONTSIZE)
    ax_left.set_ylim(bottom=0.0)
    ax_left.set_xticks([0, 5, 10, 15, 20, 25, 30])
    ax_right.set_xticks([970, 985, 1000])
    for ax in (ax_left, ax_right):
        ax.tick_params(direction="in", which="both", top=True, right=True, labelsize=TICK_FONTSIZE)
        ax.grid(alpha=0.28, linestyle=":")
    add_break_marks(ax_left, ax_right, color="black")
    fig.supxlabel(r"Site $j$", fontsize=AXIS_LABEL_FONTSIZE, y=0.03)
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=4, fontsize=LEGEND_FONTSIZE)
    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.16, top=0.82, wspace=0.08)

    out_path = OUT_DIR / "majorana_L1000.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path.relative_to(PROJECT_ROOT)}")
    return out_path


def animate_cinematic_majorana(L: int = L_DEFAULT) -> Path:
    """Render the MP4 visualization in the Majorana media style."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g_frames = timeline(FRAMES_TOPO_VIDEO, FRAMES_TRIVIAL_VIDEO)
    sites = np.arange(1, L + 1)
    marker_step = max(1, L // 45)

    plt.style.use("dark_background")
    fig, ax_left, ax_right = make_broken_axes(
        figsize=(12, 7),
        facecolor="#0B0C10",
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.05},
    )
    axes = (ax_left, ax_right)
    for ax in axes:
        ax.set_facecolor("#0B0C10")
        ax.set_ylim(0.0, 0.55)
        ax.tick_params(colors="#C5C6C7", labelsize=TICK_FONTSIZE, direction="in")
        ax.grid(color="#1F2833", linestyle=":", linewidth=0.7, alpha=0.8)
        for spine in ax.spines.values():
            spine.set_color("#1F2833")
    ax_left.set_ylabel(r"$\rho_j$", fontsize=AXIS_LABEL_FONTSIZE, color="#66FCF1", labelpad=10)
    ax_left.set_xticks([0, 15, 30])
    ax_right.set_xticks([970, 985, 1000])
    add_break_marks(ax_left, ax_right, color="#C5C6C7")
    fig.supxlabel(r"Site $j$", fontsize=AXIS_LABEL_FONTSIZE, color="#66FCF1", y=0.055)

    glow_lines = []
    scatter_pts = []
    for ax in axes:
        ax_lines = []
        for lw, alpha in [(9, 0.05), (6, 0.15), (3, 0.4), (1.5, 1.0)]:
            line, = ax.plot([], [], "-", lw=lw, alpha=alpha, solid_capstyle="round")
            ax_lines.append(line)
        glow_lines.append(ax_lines)
        scatter, = ax.plot([], [], "o", ms=5.5, color="#FFFFFF", zorder=5, markevery=marker_step)
        scatter_pts.append(scatter)
    text_g = fig.text(0.5, 0.92, "", ha="center", fontsize=AXIS_LABEL_FONTSIZE, color="#FFFFFF")
    text_xi = fig.text(0.5, 0.84, "", ha="center", fontsize=TEXT_FONTSIZE, color="#C5C6C7")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.14, top=0.80, wspace=0.05)

    def update(frame_idx: int):
        g = float(g_frames[frame_idx])
        rho = profile_density_large_l(L, g)
        neon_color = "#66FCF1" if g < 1.0 else "#FF0055"
        fill_color = "#45A29E" if g < 1.0 else "#C3073F"
        artists = []
        for ax, ax_lines, scatter in zip(axes, glow_lines, scatter_pts):
            for line in ax_lines:
                line.set_data(sites, rho)
                line.set_color(neon_color)
                artists.append(line)
            scatter.set_data(sites, rho)
            scatter.set_color(neon_color)
            artists.append(scatter)
            while ax.collections:
                ax.collections[0].remove()
            ax.fill_between(sites, 0.0, rho, color=fill_color, alpha=0.20)
        text_g.set_text(rf"OBC, $L={L}$, $g={g:.3f}$")
        text_xi.set_text(xi_label(g))
        return artists + [text_g, text_xi]

    ani = animation.FuncAnimation(fig, update, frames=TOTAL_FRAMES_VIDEO, blit=False, interval=1000 / FPS_VIDEO)
    out_path = OUT_DIR / "majorana_cinematic_60fps.mp4"
    writer = animation.FFMpegWriter(fps=FPS_VIDEO, bitrate=5000)
    ani.save(out_path, writer=writer)
    plt.close(fig)
    print(f"[OK] {out_path.relative_to(PROJECT_ROOT)}")
    return out_path


def animate_github_readme_gif(L: int = L_DEFAULT) -> Path:
    """Render the GIF version with the same layout and styling as the MP4."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g_frames = timeline(FRAMES_TOPO_GIF, FRAMES_TRIVIAL_GIF)
    sites = np.arange(1, L + 1)
    marker_step = max(1, L // 45)

    plt.style.use("dark_background")
    fig, ax_left, ax_right = make_broken_axes(
        figsize=(12, 7),
        facecolor="#0B0C10",
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.05},
    )
    fig.set_dpi(100)
    axes = (ax_left, ax_right)
    for ax in axes:
        ax.set_facecolor("#0B0C10")
        ax.set_ylim(0.0, 0.55)
        ax.tick_params(colors="#C5C6C7", labelsize=TICK_FONTSIZE, direction="in")
        ax.grid(color="#1F2833", linestyle=":", linewidth=0.7, alpha=0.8)
        for spine in ax.spines.values():
            spine.set_color("#1F2833")
    ax_left.set_ylabel(r"$\rho_j$", fontsize=AXIS_LABEL_FONTSIZE, color="#66FCF1", labelpad=10)
    ax_left.set_xticks([0, 15, 30])
    ax_right.set_xticks([970, 985, 1000])
    add_break_marks(ax_left, ax_right, color="#C5C6C7")

    glow_lines = []
    scatter_pts = []
    for ax in axes:
        ax_lines = []
        for lw, alpha in [(9, 0.05), (6, 0.15), (3, 0.4), (1.5, 1.0)]:
            line, = ax.plot([], [], "-", lw=lw, alpha=alpha, solid_capstyle="round")
            ax_lines.append(line)
        glow_lines.append(ax_lines)
        scatter, = ax.plot([], [], "o", ms=5.5, color="#FFFFFF", zorder=5, markevery=marker_step)
        scatter_pts.append(scatter)
    text_g = fig.text(0.5, 0.92, "", ha="center", fontsize=AXIS_LABEL_FONTSIZE, color="#FFFFFF")
    text_xi = fig.text(0.5, 0.84, "", ha="center", fontsize=TEXT_FONTSIZE, color="#C5C6C7")
    fig.supxlabel(r"Site $j$", fontsize=AXIS_LABEL_FONTSIZE, color="#66FCF1", y=0.055)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.14, top=0.80, wspace=0.05)

    def update(frame_idx: int):
        g = float(g_frames[frame_idx])
        rho = profile_density_large_l(L, g)
        neon_color = "#66FCF1" if g < 1.0 else "#FF0055"
        fill_color = "#45A29E" if g < 1.0 else "#C3073F"
        artists = []
        for ax, ax_lines, scatter in zip(axes, glow_lines, scatter_pts):
            for line in ax_lines:
                line.set_data(sites, rho)
                line.set_color(neon_color)
                artists.append(line)
            scatter.set_data(sites, rho)
            scatter.set_color(neon_color)
            artists.append(scatter)
            while ax.collections:
                ax.collections[0].remove()
            ax.fill_between(sites, 0.0, rho, color=fill_color, alpha=0.20)
        text_g.set_text(rf"OBC, $L={L}$, $g={g:.3f}$")
        text_xi.set_text(xi_label(g))
        return artists + [text_g, text_xi]

    ani = animation.FuncAnimation(fig, update, frames=TOTAL_FRAMES_GIF, blit=False, interval=1000 / FPS_GIF)
    out_path = OUT_DIR / "majorana_github_readme.gif"
    writer = animation.PillowWriter(fps=FPS_GIF)
    ani.save(out_path, writer=writer)
    plt.close(fig)
    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[OK] {out_path.relative_to(PROJECT_ROOT)} ({file_size_mb:.2f} MB)")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate L=1000 Majorana PDF/GIF/MP4 outputs.")
    parser.add_argument("--L", type=int, default=L_DEFAULT)
    parser.add_argument("--skip-pdf", action="store_true")
    parser.add_argument("--skip-mp4", action="store_true")
    parser.add_argument("--skip-gif", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_pdf:
        print("[majorana_L1000.py] Rendering PDF ...")
        save_pdf(L=args.L)
    if not args.skip_mp4:
        print("[majorana_L1000.py] Rendering MP4 ...")
        animate_cinematic_majorana(L=args.L)
    if not args.skip_gif:
        print("[majorana_L1000.py] Rendering GIF ...")
        animate_github_readme_gif(L=args.L)
    if args.skip_pdf and args.skip_mp4 and args.skip_gif:
        print("[majorana_L1000.py] Nothing to render.")


if __name__ == "__main__":
    main()
