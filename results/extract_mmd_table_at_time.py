from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BLOCK_SPECS = [
    ("After 10\\,s", 10.0),
    ("After 60\\,s", 60.0),
]

CELL_RUN_SPECS = [
    ("QDSB (ours)", Path("results/output_mmd_sf2m_cell_coverage.json")),
    ("DSB (2021)", Path("results/output_mmd_dsb_cell.json")),
    ("DSBM (2023)", Path("results/output_mmd_dsbm_cell.json")),
    ("SF2M (2024)", Path("results/output_mmd_sf2m_cell.json")),
    ("SF2M + mPOT", Path("results/output_mmd_sf2m_mpot_cell.json")),
    ("LightSB-M (2024)", Path("results/output_mmd_lightsb_m_cell.json")),
]

SUITE_SPECS = {
    "cell": {
        "target_epoch": 1000,
        "epoch_group_label": "1000 epochs + total time",
        "epoch_metric_label": "1000 epochs",
        "dataset_order": ["EB", "Cite", "Multi"],
        "dataset_headers": {
            "EB": "EB",
            "Cite": "Cite",
            "Multi": "Multi",
        },
        "run_specs": CELL_RUN_SPECS,
        "combined_results": None,
        "algorithm_order": None,
        "algorithm_label_map": None,
    },
    "2d": {
        "target_epoch": 500,
        "epoch_group_label": "500 epochs + total time",
        "epoch_metric_label": "500 epochs",
        "dataset_order": [
            "8Gaussians -> Moons",
            "Gaussian -> Moons",
            "Gaussian -> 8Gaussians",
        ],
        "dataset_headers": {
            "8Gaussians -> Moons": r"8G$\to$M",
            "Gaussian -> Moons": r"G$\to$M",
            "Gaussian -> 8Gaussians": r"G$\to$8G",
        },
        "run_specs": None,
        "combined_results": Path("results/output_toy_2d_suite.json"),
        "algorithm_order": [
            "qdsb",
            "dsb",
            "dsbm",
            "sf2m",
            "sf2m_mpot",
            "lightsb_m",
        ],
        "algorithm_label_map": {
            "qdsb": "QDSB (ours)",
            "dsb": "DSB (2021)",
            "dsbm": "DSBM (2023)",
            "sf2m": "SF2M (2024)",
            "sf2m_mpot": "SF2M + mPOT",
            "lightsb_m": "LightSB-M (2024)",
        },
    },
}

METHOD_LABELS = {
    "QDSB (ours)": r"\shortstack[l]{QDSB\\(ours)}",
    "DSB (2021)": r"\shortstack[l]{DSB\\(2021)}",
    "DSBM (2023)": r"\shortstack[l]{DSBM\\(2023)}",
    "SF2M (2024)": r"\shortstack[l]{SF2M\\(2024)}",
    "SF2M + mPOT": r"\shortstack[l]{SF2M\\+ mPOT}",
    "LightSB-M (2024)": r"\shortstack[l]{LightSB-M\\(2024)}",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print a LaTeX MMD table for either the cell suite or the 2D toy suite."
        ),
    )
    parser.add_argument(
        "--suite",
        choices=sorted(SUITE_SPECS),
        default="cell",
        help="Which experiment suite to summarize.",
    )
    return parser.parse_args()


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
    raise ValueError("Missing 'quality_curve_metric' in results payload.")


def extract_interpolated_metric(
    result: dict[str, object],
    *,
    metric_key: str,
    elapsed_seconds: float,
) -> tuple[float, float]:
    quality_curve = result.get("quality_curve")
    if not isinstance(quality_curve, list) or not quality_curve:
        raise ValueError(
            f"Dataset {result.get('dataset', '<unknown>')} has no quality curve."
        )

    x_key = "mean_elapsed_seconds"
    y_key = f"mean_{metric_key}"
    y_std_key = f"std_{metric_key}"
    times = np.asarray([point[x_key] for point in quality_curve], dtype=np.float64)
    values = np.asarray([point[y_key] for point in quality_curve], dtype=np.float64)
    value_std = np.asarray(
        [point.get(y_std_key, 0.0) for point in quality_curve],
        dtype=np.float64,
    )
    mean_value = float(np.interp(elapsed_seconds, times, values))
    std_value = float(np.interp(elapsed_seconds, times, value_std))
    return mean_value, std_value


def extract_metric_at_epoch(
    result: dict[str, object],
    *,
    metric_key: str,
    epoch: int,
) -> tuple[float, float]:
    quality_curve = result.get("quality_curve")
    if not isinstance(quality_curve, list) or not quality_curve:
        raise ValueError(
            f"Dataset {result.get('dataset', '<unknown>')} has no quality curve."
        )

    y_key = f"mean_{metric_key}"
    y_std_key = f"std_{metric_key}"
    for point in quality_curve:
        if int(point["epoch"]) == epoch:
            return float(point[y_key]), float(point.get(y_std_key, 0.0))

    raise ValueError(
        f"Epoch {epoch} is missing from the aggregate quality curve for "
        f"{result.get('dataset', '<unknown>')}."
    )


def extract_elapsed_time_at_epoch(
    result: dict[str, object],
    *,
    epoch: int,
) -> float:
    quality_curve = result.get("quality_curve")
    if not isinstance(quality_curve, list) or not quality_curve:
        raise ValueError(
            f"Dataset {result.get('dataset', '<unknown>')} has no quality curve."
        )

    for point in quality_curve:
        if int(point["epoch"]) == epoch:
            return float(point["mean_elapsed_seconds"])

    raise ValueError(
        f"Epoch {epoch} is missing from the aggregate quality curve for "
        f"{result.get('dataset', '<unknown>')}."
    )


def format_seconds(seconds: float) -> str:
    rounded_seconds = int(round(seconds))
    return rf"{rounded_seconds}\,s"


def format_metric_value(mean: float) -> str:
    return f"{mean:.3f}"


def format_method_label(label: str) -> str:
    try:
        return METHOD_LABELS[label]
    except KeyError as exc:
        raise ValueError(f"Unexpected method label: {label}") from exc


def maybe_bold_metric(mean: float, std: float, *, is_best: bool) -> str:
    mean_text = format_metric_value(mean)
    if is_best:
        mean_text = rf"\textbf{{{mean_text}}}"
    return rf"\shortstack[c]{{{mean_text}\\$\pm$ {std:.3f}}}"


def maybe_bold_time(seconds: float, *, is_best: bool) -> str:
    time_text = format_seconds(seconds)
    if is_best:
        time_text = rf"\textbf{{{time_text}}}"
    return rf"\multirow[c]{{1}}{{*}}[0.5em]{{{time_text}}}"


def load_suite_runs(
    suite: str,
) -> tuple[list[str], dict[str, str], list[tuple[str, dict[str, dict[str, object]]]]]:
    spec = SUITE_SPECS[suite]
    dataset_order = list(spec["dataset_order"])
    dataset_headers = dict(spec["dataset_headers"])

    if spec["run_specs"] is not None:
        loaded_runs = []
        for label, results_file in spec["run_specs"]:
            results = load_results(results_file)
            metric_key = infer_metric_key(results[0])
            if metric_key.lower() != "mmd":
                raise ValueError(
                    f"{results_file} uses quality curve metric {metric_key!r}, expected 'mmd'."
                )
            results_by_dataset = {
                str(result.get("dataset")): result
                for result in results
                if result.get("dataset") is not None
            }
            loaded_runs.append((label, results_by_dataset))
        return dataset_order, dataset_headers, loaded_runs

    combined_results_path = spec["combined_results"]
    if combined_results_path is None:
        raise ValueError(f"Suite {suite!r} is misconfigured.")

    results = load_results(combined_results_path)
    metric_key = infer_metric_key(results[0])
    if metric_key.lower() != "mmd":
        raise ValueError(
            f"{combined_results_path} uses quality curve metric {metric_key!r}, expected 'mmd'."
        )

    algorithm_order = list(spec["algorithm_order"])
    algorithm_label_map = dict(spec["algorithm_label_map"])
    results_by_algorithm_dataset: dict[str, dict[str, dict[str, object]]] = {}
    for result in results:
        algorithm_key = result.get("algorithm_key")
        dataset_name = result.get("dataset")
        if not isinstance(algorithm_key, str) or not isinstance(dataset_name, str):
            continue
        results_by_algorithm_dataset.setdefault(algorithm_key, {})[dataset_name] = result

    loaded_runs = []
    for algorithm_key in algorithm_order:
        if algorithm_key not in results_by_algorithm_dataset:
            raise ValueError(
                f"Algorithm key {algorithm_key!r} is missing from {combined_results_path}."
            )
        loaded_runs.append(
            (
                algorithm_label_map[algorithm_key],
                results_by_algorithm_dataset[algorithm_key],
            )
        )
    return dataset_order, dataset_headers, loaded_runs


def main() -> None:
    args = parse_args()
    suite_spec = SUITE_SPECS[args.suite]
    target_epoch = int(suite_spec["target_epoch"])
    epoch_group_label = str(suite_spec["epoch_group_label"])
    epoch_metric_label = str(suite_spec["epoch_metric_label"])
    dataset_order, dataset_headers, loaded_runs = load_suite_runs(args.suite)

    mean_epoch_times: dict[str, float] = {}
    row_data: dict[str, dict[str, object]] = {}

    for label, results_by_dataset in loaded_runs:
        by_block: dict[str, dict[str, tuple[float, float]]] = {}
        for block_label, elapsed_seconds in BLOCK_SPECS:
            dataset_metrics: dict[str, tuple[float, float]] = {}
            for dataset_label in dataset_order:
                result = results_by_dataset.get(dataset_label)
                if result is None:
                    raise ValueError(
                        f"Dataset {dataset_label!r} is missing for {label}."
                    )
                mean_value, std_value = extract_interpolated_metric(
                    result,
                    metric_key="mmd",
                    elapsed_seconds=elapsed_seconds,
                )
                dataset_metrics[dataset_label] = (mean_value, std_value)
            by_block[block_label] = dataset_metrics

        epoch_metrics: dict[str, tuple[float, float]] = {}
        elapsed_at_target_epoch = []
        for dataset_label in dataset_order:
            result = results_by_dataset.get(dataset_label)
            if result is None:
                raise ValueError(
                    f"Dataset {dataset_label!r} is missing for {label}."
                )
            mean_value, std_value = extract_metric_at_epoch(
                result,
                metric_key="mmd",
                epoch=target_epoch,
            )
            epoch_metrics[dataset_label] = (mean_value, std_value)
            elapsed_at_target_epoch.append(
                extract_elapsed_time_at_epoch(
                    result,
                    epoch=target_epoch,
                )
            )

        mean_epoch_times[label] = float(np.mean(elapsed_at_target_epoch))
        row_data[label] = {
            "blocks": by_block,
            "epoch_metrics": epoch_metrics,
        }

    best_metric_display_values: dict[tuple[str, str], str] = {}
    for block_label, _elapsed_seconds in BLOCK_SPECS:
        for dataset_label in dataset_order:
            min_mean = min(
                row_data[label]["blocks"][block_label][dataset_label][0]
                for label, _ in loaded_runs
            )
            best_metric_display_values[(block_label, dataset_label)] = (
                format_metric_value(min_mean)
            )

    for dataset_label in dataset_order:
        min_mean = min(
            row_data[label]["epoch_metrics"][dataset_label][0]
            for label, _ in loaded_runs
        )
        best_metric_display_values[(epoch_metric_label, dataset_label)] = (
            format_metric_value(min_mean)
        )

    best_time_display = format_seconds(min(mean_epoch_times.values()))

    print(r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}Xccc|ccc|cccc@{}}")
    print(r"\toprule")
    print(
        r"\multirow[c]{2}{*}[-0.8mm]{Method} "
        r"& \multicolumn{3}{c}{After 10\,s} "
        r"& \multicolumn{3}{c}{After 60\,s} "
        rf"& \multicolumn{{4}}{{c}}{{{epoch_group_label}}} \\"
    )
    print(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(ll){8-11}")
    print(
        "& "
        + " & ".join(dataset_headers[name] for name in dataset_order)
        + " & "
        + " & ".join(dataset_headers[name] for name in dataset_order)
        + " & "
        + " & ".join(dataset_headers[name] for name in dataset_order)
        + r" & Time \\"
    )
    print(r"\midrule")

    for row_idx, (label, _results_by_dataset) in enumerate(loaded_runs):
        cells = []
        blocks = row_data[label]["blocks"]
        assert isinstance(blocks, dict)
        for block_label, _elapsed_seconds in BLOCK_SPECS:
            dataset_metrics = blocks[block_label]
            assert isinstance(dataset_metrics, dict)
            for dataset_label in dataset_order:
                mean_value, std_value = dataset_metrics[dataset_label]
                cells.append(
                    maybe_bold_metric(
                        mean_value,
                        std_value,
                        is_best=format_metric_value(mean_value)
                        == best_metric_display_values[(block_label, dataset_label)],
                    )
                )

        epoch_metrics = row_data[label]["epoch_metrics"]
        assert isinstance(epoch_metrics, dict)
        for dataset_label in dataset_order:
            mean_value, std_value = epoch_metrics[dataset_label]
            cells.append(
                maybe_bold_metric(
                    mean_value,
                    std_value,
                    is_best=format_metric_value(mean_value)
                    == best_metric_display_values[(epoch_metric_label, dataset_label)],
                )
            )

        cells.append(
            maybe_bold_time(
                mean_epoch_times[label],
                is_best=format_seconds(mean_epoch_times[label]) == best_time_display,
            )
        )

        line_suffix = r" \\[0.5em]" if row_idx < len(loaded_runs) - 1 else r" \\"
        print(format_method_label(label) + " & " + " & ".join(cells) + line_suffix)

    print(r"\bottomrule")
    print(r"\end{tabularx}")


if __name__ == "__main__":
    main()
