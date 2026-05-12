from __future__ import annotations

import argparse
import json
import math
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

EMA_ALPHA = 0.2
DEFAULT_INPUT = Path("results/output_toy_2d_suite.json")
DEFAULT_DATASET_ORDER = [
    "8Gaussians -> Moons",
    "Gaussian -> Moons",
    "Gaussian -> 8Gaussians",
]
ALGORITHM_ORDER = {
    "qdsb": (0, "QDSB (ours)"),
    "dsb": (1, "DSB"),
    "dsbm": (2, "DSBM"),
    "lightsb_m": (3, "LightSB-M"),
    "sf2m": (4, "SF2M"),
    "sf2m_mpot": (5, "SF2M + mPOT"),
    "trajectory_net": (6, "TrajectoryNet"),
}
LINE_SPACING = 0.6
LINE_STYLES = {
    "QDSB (ours)": "-",
    "DSB": (0, (1.0, LINE_SPACING)),
    "DSBM": (0, (4.0, LINE_SPACING, 1.2, LINE_SPACING, 1.2, LINE_SPACING)),
    "LightSB-M": (0, (4.0, LINE_SPACING, 4.0, LINE_SPACING, 1.2, LINE_SPACING)),
    "SF2M": (0, (5.0, LINE_SPACING)),
    "SF2M + mPOT": (0, (2.5, 0.75, 1.0, 0.75)),
    "TrajectoryNet": (0, (2.0, LINE_SPACING, 2.0, LINE_SPACING)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot elapsed time versus MMD for the 2D toy benchmark suite "
            "from results/output_toy_2d_suite.json."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Toy-suite JSON file to plot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Directory where the generated PDF should be written.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="output_toy_time_vs_mmd.pdf",
        help="Filename for the combined PDF plot.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASET_ORDER,
        help="Datasets to include in the plot, in display order.",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=None,
        help="Optional common maximum elapsed time in seconds.",
    )
    return parser.parse_args()


def load_results(input_path: Path) -> list[dict[str, object]]:
    with input_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        results = payload.get("results")
        if not isinstance(results, list):
            raise ValueError(
                f"{input_path} does not contain a top-level 'results' list."
            )
        return results

    if isinstance(payload, list):
        return payload

    raise ValueError(f"Unsupported JSON structure in {input_path}.")


def infer_metric_key(result: dict[str, object]) -> str:
    metric_key = result.get("quality_curve_metric")
    if isinstance(metric_key, str) and metric_key:
        return metric_key

    quality_curve = result.get("quality_curve")
    if not isinstance(quality_curve, list) or not quality_curve:
        raise ValueError("Missing 'quality_curve' data.")

    first_entry = quality_curve[0]
    for key in first_entry:
        if key.startswith("mean_"):
            return key.removeprefix("mean_")
    raise ValueError("Could not infer quality curve metric.")


def metric_label(metric_key: str) -> str:
    if metric_key.lower() == "w1":
        return "W1"
    if metric_key.lower() == "mmd":
        return "MMD"
    return metric_key.replace("_", " ").title()


def extract_curve(
    result: dict[str, object],
    *,
    metric_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    quality_curve = result.get("quality_curve")
    if not isinstance(quality_curve, list) or not quality_curve:
        raise ValueError(
            f"Dataset {result.get('dataset', '<unknown>')} has no quality curve."
        )

    x_key = "mean_elapsed_seconds"
    y_key = f"mean_{metric_key}"
    times = np.asarray([point[x_key]
                       for point in quality_curve], dtype=np.float64)
    values = np.asarray([point[y_key]
                        for point in quality_curve], dtype=np.float64)
    return times, values


def ema_smooth(values: np.ndarray, *, alpha: float = EMA_ALPHA) -> np.ndarray:
    if values.ndim != 1:
        raise ValueError("EMA smoothing expects a 1D array.")
    if values.size == 0:
        return values.copy()

    smoothed = values.astype(np.float64, copy=True)
    for idx in range(1, smoothed.size):
        smoothed[idx] = alpha * smoothed[idx] + \
            (1.0 - alpha) * smoothed[idx - 1]
    return smoothed


def round_up_nice(value: float) -> float:
    if value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    base = 10 ** exponent
    for multiplier in (1.0, 2.0, 5.0, 10.0):
        candidate = multiplier * base
        if candidate >= value:
            return candidate
    return 10.0 * base


def compute_ylim(series: list[tuple[np.ndarray, np.ndarray]]) -> tuple[float, float]:
    all_values = np.concatenate([ema_smooth(values) for _, values in series])
    ymin = float(np.min(all_values))
    ymax = float(np.max(all_values))
    if math.isclose(ymin, ymax):
        pad = max(0.01 * abs(ymin), 1e-3)
        return ymin - pad, ymax + pad
    pad = 0.06 * (ymax - ymin)
    return ymin - pad, ymax + pad


def main() -> None:
    args = parse_args()
    results = load_results(args.input)
    metric_key = infer_metric_key(results[0])

    for result in results[1:]:
        other_metric_key = infer_metric_key(result)
        if other_metric_key != metric_key:
            raise ValueError(
                "All toy-suite results must use the same quality curve metric, got "
                f"{metric_key!r} and {other_metric_key!r}."
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results_by_dataset_algorithm: dict[str, dict[str, dict[str, object]]] = {}
    for result in results:
        dataset = str(result.get("dataset"))
        algorithm_key = str(result.get("algorithm_key"))
        results_by_dataset_algorithm.setdefault(
            dataset, {})[algorithm_key] = result

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

    fig, axes = plt.subplots(
        1,
        len(args.datasets),
        figsize=(5.35, 1.9),
        squeeze=False,
    )
    axes_row = axes[0]
    legend_handles = None
    legend_labels = None

    for ax, dataset_label in zip(axes_row, args.datasets):
        dataset_results = results_by_dataset_algorithm.get(dataset_label)
        if dataset_results is None:
            raise ValueError(
                f"Dataset {dataset_label!r} is missing from {args.input}.")

        series: list[tuple[np.ndarray, np.ndarray, str]] = []
        for algorithm_key, (sort_idx, label) in sorted(ALGORITHM_ORDER.items(), key=lambda item: item[1][0]):
            result = dataset_results.get(algorithm_key)
            if result is None:
                continue
            del sort_idx
            times, values = extract_curve(result, metric_key=metric_key)
            series.append((times, values, label))

        if not series:
            raise ValueError(
                f"Dataset {dataset_label!r} in {args.input} contains no plottable series."
            )

        for times, values, label in series:
            ax.plot(
                times,
                ema_smooth(values),
                linestyle=LINE_STYLES.get(label, "-"),
                linewidth=2.0,
                label=label,
            )

        if legend_handles is None or legend_labels is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

        ax.set_xlabel("Elapsed Time (s)", fontsize=fs, labelpad=1)
        if dataset_label == args.datasets[0]:
            ax.set_ylabel(metric_label(metric_key), fontsize=fs, labelpad=1)
        # ax.set_title(dataset_label, fontsize=fs, pad=2)
        ax.grid(True, alpha=0.3)
        if dataset_label == args.datasets[0]:
            ax.set_xlim(0, 20)
            ax.set_xticks([0, 5, 10, 15, 20])
            ax.set_ylim(0.0, 0.2)
            ax.set_yticks([0.0, 0.05, 0.1, 0.15, 0.2])
        elif dataset_label == args.datasets[1]:
            ax.set_xlim(0, 20)
            ax.set_xticks([0, 5, 10, 15, 20])
            ax.set_ylim(0.0, 0.2)
            ax.set_yticks([0.0, 0.05, 0.1, 0.15, 0.2])
        elif dataset_label == args.datasets[2]:
            ax.set_xlim(0, 20)
            ax.set_xticks([0, 5, 10, 15, 20])
            ax.set_ylim(0.0, 0.2)
            ax.set_yticks([0.0, 0.05, 0.1, 0.15, 0.2])
        ax.tick_params(axis="both", labelsize=fs, pad=1)

    if legend_handles is not None and legend_labels is not None:
        legend = fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.97),
            ncol=3,
            fontsize=fs,
            frameon=False,
            columnspacing=1.1,
            handlelength=2.2,
        )
        legend_artists = getattr(
            legend,
            "legend_handles",
            getattr(legend, "legendHandles", []),
        )
        for artist in legend_artists:
            if hasattr(artist, "set_linewidth"):
                artist.set_linewidth(1.8)
            if hasattr(artist, "set_markersize"):
                artist.set_markersize(4.0)

    fig.subplots_adjust(
        left=0.09,
        right=0.995,
        top=0.98,
        bottom=0.36,
        wspace=0.25,
    )
    output_path = args.output_dir / args.output_name
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
