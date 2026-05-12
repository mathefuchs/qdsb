from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

MPLCONFIGDIR = Path(tempfile.gettempdir()) / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPLCONFIGDIR))
matplotlib.use("Agg")

DEFAULT_DATASET_ORDER = ["EB", "Cite", "Multi"]
EMA_ALPHA = 0.2
DEFAULT_RESULTS_GLOB = "output_mmd_*.json"
RUN_LABELS = {
    "output_mmd_sf2m_cell_coverage": (0, "QDSB (ours)"),
    "output_mmd_dsb_cell": (1, "DSB"),
    "output_mmd_dsbm_cell": (2, "DSBM"),
    "output_mmd_lightsb_m_cell": (3, "LightSB-M"),
    "output_mmd_sf2m_cell": (4, "SF2M"),
    "output_mmd_sf2m_mpot_cell": (5, "SF2M + mPOT"),
}
LINE_SPACING = 0.6
LINE_STYLES = {
    "QDSB (ours)": "-",
    "DSB": (0, (1.0, LINE_SPACING)),
    "DSBM": (0, (4.0, LINE_SPACING, 1.2, LINE_SPACING, 1.2, LINE_SPACING)),
    "LightSB-M": (0, (4.0, LINE_SPACING, 4.0, LINE_SPACING, 1.2, LINE_SPACING)),
    "SF2M": (0, (5.0, LINE_SPACING)),
    "SF2M + mPOT": (0, (2.5, 0.75, 1.0, 0.75)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a single combined elapsed-time versus MMD figure for the "
            "available cell experiment MMD result files."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing the output_mmd_*.json files to plot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Directory where the generated PDF plots should be written.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="output_time_vs_mmd.pdf",
        help="Filename for the combined PDF plot.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASET_ORDER,
        help="Datasets to include in the plot, in display order.",
    )
    return parser.parse_args()


def discover_runs(results_dir: Path) -> list[tuple[str, Path]]:
    results_files = sorted(results_dir.glob(DEFAULT_RESULTS_GLOB))
    if not results_files:
        raise ValueError(
            f"No files matching {DEFAULT_RESULTS_GLOB!r} were found in {results_dir}."
        )

    runs = []
    for results_file in results_files:
        stem = results_file.stem
        idx, label = RUN_LABELS[stem]
        runs.append((idx, label, results_file))
    return list(map(lambda t: (t[1], t[2]), sorted(runs)))


def load_results(results_file: Path) -> list[dict[str, object]]:
    with results_file.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        results = payload.get("results")
        if not isinstance(results, list):
            raise ValueError(
                f"{results_file} does not contain a top-level 'results' list."
            )
        return results

    if isinstance(payload, list):
        return payload

    raise ValueError(f"Unsupported JSON structure in {results_file}.")


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    quality_curve = result.get("quality_curve")
    if not isinstance(quality_curve, list) or not quality_curve:
        raise ValueError(
            f"Dataset {result.get('dataset', '<unknown>')} has no quality curve."
        )

    x_key = "mean_elapsed_seconds"
    y_key = f"mean_{metric_key}"
    y_std_key = f"std_{metric_key}"

    times = np.asarray([point[x_key]
                       for point in quality_curve], dtype=np.float64)
    values = np.asarray([point[y_key]
                        for point in quality_curve], dtype=np.float64)
    value_std = np.asarray(
        [point.get(y_std_key, 0.0) for point in quality_curve],
        dtype=np.float64,
    )
    return times, values, value_std


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


def plot_curve(
    ax: plt.Axes,
    *,
    label: str,
    times: np.ndarray,
    values: np.ndarray,
    value_std: np.ndarray,
) -> None:
    smoothed_values = ema_smooth(values)
    # smoothed_std = ema_smooth(value_std)

    ax.plot(
        times,
        smoothed_values,
        linestyle=LINE_STYLES.get(label, "-"),
        linewidth=2.0,
        label=label,
    )
    # if np.any(smoothed_std > 0):
    #     ax.fill_between(
    #         times,
    #         smoothed_values - smoothed_std,
    #         smoothed_values + smoothed_std,
    #         alpha=0.05,
    #     )


def main() -> None:
    args = parse_args()
    run_specs = discover_runs(args.results_dir)

    loaded_runs = []
    for label, results_file in run_specs:
        results = load_results(results_file)
        results_by_dataset = {
            str(result.get("dataset")): result
            for result in results
            if result.get("dataset") is not None
        }
        loaded_runs.append(
            {
                "path": results_file,
                "label": label,
                "metric_key": infer_metric_key(results[0]),
                "results_by_dataset": results_by_dataset,
            }
        )

    metric_key = str(loaded_runs[0]["metric_key"])
    for run in loaded_runs[1:]:
        other_metric_key = str(run["metric_key"])
        if other_metric_key != metric_key:
            raise ValueError(
                "All datasets in all results files must use the same quality "
                f"curve metric, got {metric_key!r} and {other_metric_key!r}."
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)

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
    plotted_dataset_count = 0
    for ax, dataset_label in zip(axes_row, args.datasets):
        series = []
        for run in loaded_runs:
            result = run["results_by_dataset"].get(dataset_label)
            if result is None:
                raise ValueError(
                    f"Dataset {dataset_label!r} is missing from {run['path']}."
                )
            times, values, value_std = extract_curve(
                result, metric_key=metric_key)
            series.append((times, values, value_std, str(run["label"])))

        for times, values, value_std, label in series:
            plot_curve(
                ax,
                label=label,
                times=times,
                values=values,
                value_std=value_std,
            )

        # ax.set_title(dataset_label)
        ax.set_xlabel("Elapsed Time (s)", fontsize=fs, labelpad=1)
        if dataset_label == "EB":
            ax.set_ylabel(metric_label(metric_key), fontsize=fs, labelpad=1)
        ax.grid(True, alpha=0.3)
        if dataset_label == "EB":
            ax.set_xlim(0, 20)
            ax.set_xticks([0, 5, 10, 15, 20])
            ax.set_ylim(0.19, 0.27)
            ax.set_yticks([0.19, 0.21, 0.23, 0.25, 0.27])
        elif dataset_label == "Cite":
            ax.set_xlim(0, 20)
            ax.set_xticks([0, 5, 10, 15, 20])
            ax.set_ylim(0.18, 0.22)
            ax.set_yticks([0.18, 0.19, 0.2, 0.21, 0.22])
        elif dataset_label == "Multi":
            ax.set_xlim(0, 20)
            ax.set_xticks([0, 5, 10, 15, 20])
            ax.set_ylim(0.21, 0.25)
            ax.set_yticks([0.21, 0.22, 0.23, 0.24, 0.25])
        ax.tick_params(axis="both", labelsize=fs, pad=1)
        plotted_dataset_count += 1

    if plotted_dataset_count == 0:
        raise ValueError("No datasets were plotted.")

    fig.subplots_adjust(
        left=0.09,
        right=0.995,
        top=0.98,
        bottom=0.36,
        wspace=0.25,
    )
    combined_output_path = args.output_dir / args.output_name
    fig.savefig(combined_output_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"Saved {combined_output_path}")


if __name__ == "__main__":
    main()
