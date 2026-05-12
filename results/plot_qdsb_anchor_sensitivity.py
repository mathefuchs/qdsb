from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import matplotlib

MPLCONFIGDIR = Path(tempfile.gettempdir()) / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPLCONFIGDIR))
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter  # noqa: E402

DEFAULT_INPUT = Path("results/output_toy_qdsb_anchor_sensitivity.json")
DEFAULT_OUTPUT = Path("plots/output_toy_qdsb_anchor_sensitivity.pdf")
DEFAULT_RADIUS_STAT = "median"
ENDPOINT_DISTANCE_MEDIAN = 5.580317497253418


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot QDSB anchor-count sensitivity on the 8Gaussians -> Moons toy setting."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Sensitivity JSON file to plot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PDF path.",
    )
    parser.add_argument(
        "--radius-stat",
        choices=["mean", "median"],
        default=DEFAULT_RADIUS_STAT,
        help="Which aggregated coverage-radius statistic to plot.",
    )
    return parser.parse_args()


def load_results(input_path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    with input_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"{input_path} must contain a top-level object.")
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError(
            f"{input_path} does not contain a non-empty 'results' list.")
    return payload.get("config", {}), results


def extract_arrays(
    results: list[dict[str, object]],
    *,
    radius_stat: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered = sorted(results, key=lambda item: int(item["coverage_anchors"]))
    anchors = np.asarray([int(item["coverage_anchors"])
                         for item in ordered], dtype=np.float64)
    mmd = np.asarray([float(item["mean_mmd"])
                     for item in ordered], dtype=np.float64)
    radius_key = (
        "mean_mean_assignment_radius"
        if radius_stat == "mean"
        else "mean_median_assignment_radius"
    )
    radius = np.asarray([float(item[radius_key])
                        for item in ordered], dtype=np.float64)
    return anchors, mmd, radius


def make_segments(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    return np.concatenate([points[:-1], points[1:]], axis=1)


def add_gradient_line(
    ax: plt.Axes,
    *,
    x: np.ndarray,
    y: np.ndarray,
    c: np.ndarray,
    norm: LogNorm,
    cmap: str,
    linewidth: float = 2.0,
    marker_size: float = 22.0,
) -> None:
    segments = make_segments(x, y)
    segment_values = np.sqrt(c[:-1] * c[1:])
    line = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidths=linewidth,
        capstyle="round",
        joinstyle="round",
    )
    line.set_array(segment_values)
    ax.add_collection(line)
    ax.scatter(
        x,
        y,
        c=c,
        cmap=cmap,
        norm=norm,
        s=marker_size,
        edgecolors="none",
        zorder=3,
    )


def set_anchor_axis_ticks(ax: plt.Axes, anchors: np.ndarray) -> None:
    tick_candidates = [1, 4, 16, 64, 256, 1024]
    tick_values = [tick for tick in tick_candidates if anchors.min()
                   <= tick <= anchors.max()]
    if anchors.min() not in tick_values:
        tick_values = [int(anchors.min())] + tick_values
    if anchors.max() not in tick_values:
        tick_values = tick_values + [int(anchors.max())]
    tick_values = sorted(set(tick_values))
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(tick_values))
    ax.xaxis.set_major_formatter(FixedFormatter(
        [str(tick) for tick in tick_values]))
    ax.xaxis.set_minor_formatter(NullFormatter())


def main() -> None:
    args = parse_args()
    _config, results = load_results(args.input)
    anchors, mmd, radius = extract_arrays(
        results, radius_stat=args.radius_stat)
    radius = radius / ENDPOINT_DISTANCE_MEDIAN

    args.output.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("tableau-colorblind10")

    fs = 9
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Times"],
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts": False,
            "text.latex.preamble": r"\usepackage{times}\usepackage{helvet}\usepackage{courier}",
            "savefig.bbox": "tight",
        }
    )
    plt.rc("xtick", labelsize=fs)
    plt.rc("ytick", labelsize=fs)

    cmap = "viridis"
    norm = LogNorm(vmin=float(anchors.min()), vmax=float(anchors.max()))

    fig = plt.figure(figsize=(5.189, 1.5))
    grid = fig.add_gridspec(
        1,
        7,
        width_ratios=[1.0, 0.1, 1.0, 0.05, 1.0, -0.25, 0.055],
        wspace=0.35,
    )
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 2])
    ax3 = fig.add_subplot(grid[0, 4])
    cax = fig.add_subplot(grid[0, 6])

    add_gradient_line(ax1, x=anchors, y=mmd, c=anchors, norm=norm, cmap=cmap)
    add_gradient_line(ax2, x=radius, y=mmd, c=anchors, norm=norm, cmap=cmap)
    add_gradient_line(ax3, x=anchors, y=radius,
                      c=anchors, norm=norm, cmap=cmap)

    set_anchor_axis_ticks(ax1, anchors)
    set_anchor_axis_ticks(ax3, anchors)

    radius_label = "Mean Coverage Radius" if args.radius_stat == "mean" else "Med.\\ Coverage Radius"

    ax1.set_xlabel(r"Anchors $k$", fontsize=fs, labelpad=1)
    ax1.set_ylabel("MMD", fontsize=fs, labelpad=1)
    ax1.set_ylim(0.00, 0.05)
    ax1.set_yticks([0.0, 0.01, 0.02, 0.03, 0.04, 0.05])
    ax1.set_xlim(anchors.min(), anchors.max())

    ax2.set_xlabel(radius_label, fontsize=fs, labelpad=1)
    ax2.set_ylabel("MMD", fontsize=fs, labelpad=1)
    radius_max = float(radius.max())
    radius_limit = max(1.0, np.ceil(radius_max * 10.0) / 10.0)
    radius_ticks = np.linspace(0.0, radius_limit, num=6)
    ax2.set_xlim(0, radius_limit)
    ax2.set_xticks(radius_ticks)
    ax2.set_ylim(0.00, 0.05)
    ax2.set_yticks([0, 0.01, 0.02, 0.03, 0.04, 0.05])

    ax3.set_xlabel(r"Anchors $k$", fontsize=fs, labelpad=1)
    ax3.set_ylabel(radius_label, fontsize=fs, labelpad=1)
    ax3.set_ylim(0, radius_limit)
    ax3.set_yticks(radius_ticks)
    ax3.set_xlim(anchors.min(), anchors.max())

    for ax in (ax1, ax2, ax3):
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=fs, pad=1)

    # mmd_pad = 0.08 * max(float(mmd.max() - mmd.min()), 1e-4)
    # ax1.set_ylim(float(mmd.min() - mmd_pad), float(mmd.max() + mmd_pad))
    # ax2.set_ylim(float(mmd.min() - mmd_pad), float(mmd.max() + mmd_pad))

    # radius_pad = 0.08 * max(float(radius.max() - radius.min()), 1e-4)
    # ax2.set_xlim(float(radius.min() - radius_pad), float(radius.max() + radius_pad))
    # ax3.set_ylim(float(radius.min() - radius_pad), float(radius.max() + radius_pad))

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    colorbar = fig.colorbar(
        sm,
        cax=cax,
        orientation="vertical",
    )
    colorbar.set_label(r"Anchors $k$", fontsize=fs, labelpad=-7, rotation=270)
    colorbar.set_ticks([float(anchors.min()), float(anchors.max())])
    colorbar.set_ticklabels([str(int(anchors.min())), str(int(anchors.max()))])
    colorbar.ax.tick_params(labelsize=fs, pad=1)

    fig.subplots_adjust(
        left=0.09,
        right=0.975,
        top=0.98,
        bottom=0.20,
    )
    fig.savefig(args.output, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
