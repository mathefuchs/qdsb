from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from toy_exp.common import format_metric, resolve_device, set_seed
from toy_exp.data import TOY_DATASETS, sample_toy_problem
from toy_exp.quality import (
    DEFAULT_QUALITY_CURVE_EVAL_POINTS,
    QUALITY_CURVE_METRIC,
    QualityCheckpointRecorder,
    compute_mmd,
    compute_w1,
    estimate_mmd_gamma,
    summarize_quality_curve,
)
from toy_exp.sf2m import CoverageAccelerationPlan, sample_predicted_target, train_pairwise_qdsb

DEFAULT_DATASET_KEY = "8gaussians_moons"
DEFAULT_ANCHOR_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
DEFAULT_OUTPUT_JSON = Path("results/output_toy_qdsb_anchor_sensitivity.json")
ALGORITHM_KEY = "qdsb"
ALGORITHM_LABEL = "QDSB (ours)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a QDSB anchor-count sensitivity sweep on the 8Gaussians -> "
            "Moons 2D toy setting."
        ),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--anchor-counts",
        nargs="+",
        type=int,
        default=list(DEFAULT_ANCHOR_COUNTS),
        help="Anchor counts to evaluate. k=1 corresponds to random single-anchor selection.",
    )
    parser.add_argument("--num-samples", type=int, default=16384)
    parser.add_argument("--num-eval-samples", type=int, default=4096)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument(
        "--ot-method",
        type=str,
        default="exact",
        choices=["exact", "sinkhorn"],
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument(
        "--quality-eval-every",
        type=int,
        default=0,
        help="Record untimed MMD checkpoints every N epochs. Use 0 to disable.",
    )
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--steps-per-unit", type=int, default=100)
    parser.add_argument("--rollout-batch-size", type=int, default=2048)
    parser.add_argument(
        "--max-eval-points",
        type=int,
        default=0,
        help="Subsample cap for final W1/MMD evaluation. Use 0 for full populations.",
    )
    parser.add_argument(
        "--w1-method",
        type=str,
        default="exact",
        choices=["exact", "sinkhorn"],
    )
    parser.add_argument("--w1-reg", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)

    parser.add_argument(
        "--coverage-anchor-selection",
        type=str,
        default="gon",
        choices=["gon", "gon_plus"],
    )
    parser.add_argument("--coverage-anchor-gon-plus-candidates", type=int, default=5)
    parser.add_argument("--coverage-anchor-weight-temperature", type=float, default=1.0)
    parser.add_argument("--coverage-anchor-refresh-epochs", type=int, default=100)
    parser.add_argument(
        "--coverage-point-match-mode",
        type=str,
        default="random",
        choices=["random", "ot"],
    )
    return parser.parse_args()


def assignment_radius_stats(
    dataset: torch.Tensor,
    *,
    anchors: torch.Tensor,
    assignments: torch.Tensor,
) -> dict[str, float]:
    nearest_deltas = dataset - anchors[assignments]
    nearest_distances = torch.linalg.vector_norm(nearest_deltas, dim=1)
    return {
        "mean": float(nearest_distances.mean().item()),
        "median": float(nearest_distances.median().item()),
    }


def run_single_seed(
    *,
    train_source: torch.Tensor,
    train_target: torch.Tensor,
    eval_source: torch.Tensor,
    eval_target: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    coverage_anchors: int,
) -> dict[str, object]:
    quality_mmd_gamma = estimate_mmd_gamma(eval_target)
    checkpoint_recorder = QualityCheckpointRecorder()
    coverage_refresh_stats: list[dict[str, float | int]] = []

    def checkpoint_mmd(predicted: torch.Tensor, *, epoch: int) -> None:
        checkpoint_recorder.time_evaluation(
            epoch=epoch,
            evaluate=lambda: compute_mmd(
                predicted,
                eval_target,
                gamma=quality_mmd_gamma,
                max_points=DEFAULT_QUALITY_CURVE_EVAL_POINTS,
            ),
        )

    set_seed(seed)

    def callback(epoch: int, flow_model, score_model) -> None:
        checkpoint_mmd(
            sample_predicted_target(
                flow_model,
                score_model,
                eval_source,
                sigma=args.sigma,
                steps_per_unit=args.steps_per_unit,
                rollout_batch_size=args.rollout_batch_size,
                device=device,
            ),
            epoch=epoch,
        )

    def coverage_refresh_callback(epoch: int, coverage_plan: CoverageAccelerationPlan) -> None:
        def evaluate() -> dict[str, float | int]:
            source_radius_stats = assignment_radius_stats(
                train_source,
                anchors=coverage_plan.source_anchors,
                assignments=coverage_plan.source_assignments,
            )
            target_radius_stats = assignment_radius_stats(
                train_target,
                anchors=coverage_plan.target_anchors,
                assignments=coverage_plan.target_assignments,
            )
            return {
                "epoch": int(epoch),
                "source_mean_assignment_radius": source_radius_stats["mean"],
                "source_median_assignment_radius": source_radius_stats["median"],
                "target_mean_assignment_radius": target_radius_stats["mean"],
                "target_median_assignment_radius": target_radius_stats["median"],
                "mean_assignment_radius": 0.5
                * (source_radius_stats["mean"] + target_radius_stats["mean"]),
                "median_assignment_radius": 0.5
                * (source_radius_stats["median"] + target_radius_stats["median"]),
            }

        coverage_refresh_stats.append(
            checkpoint_recorder.time_block(evaluate)
        )

    flow_model, score_model = train_pairwise_qdsb(
        train_source,
        train_target,
        coverage_anchors=coverage_anchors,
        coverage_anchor_selection=args.coverage_anchor_selection,
        coverage_anchor_gon_plus_candidates=args.coverage_anchor_gon_plus_candidates,
        coverage_anchor_weight_temperature=args.coverage_anchor_weight_temperature,
        coverage_anchor_refresh_epochs=args.coverage_anchor_refresh_epochs,
        coverage_point_match_mode=args.coverage_point_match_mode,
        sigma=args.sigma,
        ot_method=args.ot_method,
        batch_size=args.batch_size,
        epochs=args.epochs,
        width=args.width,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        progress_label=(
            f"{ALGORITHM_LABEL} {TOY_DATASETS[DEFAULT_DATASET_KEY].label} "
            f"k={coverage_anchors} seed={seed}"
        ),
        quality_eval_every=args.quality_eval_every,
        checkpoint_callback=callback if args.quality_eval_every > 0 else None,
        coverage_refresh_callback=coverage_refresh_callback,
    )
    elapsed_seconds = checkpoint_recorder.elapsed_seconds()
    predicted = sample_predicted_target(
        flow_model,
        score_model,
        eval_source,
        sigma=args.sigma,
        steps_per_unit=args.steps_per_unit,
        rollout_batch_size=args.rollout_batch_size,
        device=device,
    )

    max_eval_points = None if args.max_eval_points <= 0 else args.max_eval_points
    w1 = compute_w1(
        predicted,
        eval_target,
        max_points=max_eval_points,
        method=args.w1_method,
        reg=args.w1_reg,
    )
    mmd = compute_mmd(
        predicted,
        eval_target,
        gamma=quality_mmd_gamma,
        max_points=max_eval_points,
    )
    source_mean_radius_values = np.asarray(
        [entry["source_mean_assignment_radius"] for entry in coverage_refresh_stats],
        dtype=np.float64,
    )
    source_median_radius_values = np.asarray(
        [entry["source_median_assignment_radius"] for entry in coverage_refresh_stats],
        dtype=np.float64,
    )
    target_mean_radius_values = np.asarray(
        [entry["target_mean_assignment_radius"] for entry in coverage_refresh_stats],
        dtype=np.float64,
    )
    target_median_radius_values = np.asarray(
        [entry["target_median_assignment_radius"] for entry in coverage_refresh_stats],
        dtype=np.float64,
    )
    mean_radius_values = np.asarray(
        [entry["mean_assignment_radius"] for entry in coverage_refresh_stats],
        dtype=np.float64,
    )
    median_radius_values = np.asarray(
        [entry["median_assignment_radius"] for entry in coverage_refresh_stats],
        dtype=np.float64,
    )

    return {
        "seed": seed,
        "elapsed_seconds": float(elapsed_seconds),
        "w1": float(w1),
        "mmd": float(mmd),
        "num_coverage_refreshes": len(coverage_refresh_stats),
        "coverage_refresh_stats": coverage_refresh_stats,
        "source_mean_assignment_radius": float(source_mean_radius_values.mean()),
        "source_median_assignment_radius": float(source_median_radius_values.mean()),
        "target_mean_assignment_radius": float(target_mean_radius_values.mean()),
        "target_median_assignment_radius": float(target_median_radius_values.mean()),
        "mean_assignment_radius": float(mean_radius_values.mean()),
        "median_assignment_radius": float(median_radius_values.mean()),
        "quality_curve": checkpoint_recorder.checkpoints,
    }


def benchmark_anchor_count(
    coverage_anchors: int,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    spec = TOY_DATASETS[DEFAULT_DATASET_KEY]
    seed_results = []
    for seed in args.seeds:
        set_seed(seed)
        samples = sample_toy_problem(
            spec,
            num_samples=args.num_samples,
            num_eval_samples=args.num_eval_samples,
        )
        seed_result = run_single_seed(
            train_source=samples["train_source"],
            train_target=samples["train_target"],
            eval_source=samples["eval_source"],
            eval_target=samples["eval_target"],
            args=args,
            device=device,
            seed=seed,
            coverage_anchors=coverage_anchors,
        )
        seed_results.append(seed_result)
        print(
            f"    seed {seed}: "
            f"W1={seed_result['w1']:.6f} | "
            f"MMD={seed_result['mmd']:.6f} | "
            f"r_mean={seed_result['mean_assignment_radius']:.6f} | "
            f"r_med={seed_result['median_assignment_radius']:.6f} | "
            f"refreshes={seed_result['num_coverage_refreshes']}"
        )

    w1_values = np.asarray([entry["w1"] for entry in seed_results], dtype=np.float64)
    mmd_values = np.asarray([entry["mmd"] for entry in seed_results], dtype=np.float64)
    elapsed_values = np.asarray(
        [entry["elapsed_seconds"] for entry in seed_results],
        dtype=np.float64,
    )
    source_mean_radius_values = np.asarray(
        [entry["source_mean_assignment_radius"] for entry in seed_results],
        dtype=np.float64,
    )
    source_median_radius_values = np.asarray(
        [entry["source_median_assignment_radius"] for entry in seed_results],
        dtype=np.float64,
    )
    target_mean_radius_values = np.asarray(
        [entry["target_mean_assignment_radius"] for entry in seed_results],
        dtype=np.float64,
    )
    target_median_radius_values = np.asarray(
        [entry["target_median_assignment_radius"] for entry in seed_results],
        dtype=np.float64,
    )
    mean_radius_values = np.asarray(
        [entry["mean_assignment_radius"] for entry in seed_results],
        dtype=np.float64,
    )
    median_radius_values = np.asarray(
        [entry["median_assignment_radius"] for entry in seed_results],
        dtype=np.float64,
    )
    return {
        "algorithm": ALGORITHM_LABEL,
        "algorithm_key": ALGORITHM_KEY,
        "dataset": spec.label,
        "dataset_key": DEFAULT_DATASET_KEY,
        "source": spec.source_name,
        "target": spec.target_name,
        "coverage_anchors": coverage_anchors,
        "coverage_anchor_selection": args.coverage_anchor_selection,
        "coverage_anchor_gon_plus_candidates": args.coverage_anchor_gon_plus_candidates,
        "coverage_anchor_weight_temperature": args.coverage_anchor_weight_temperature,
        "coverage_anchor_refresh_epochs": args.coverage_anchor_refresh_epochs,
        "coverage_point_match_mode": args.coverage_point_match_mode,
        "num_samples": args.num_samples,
        "num_eval_samples": args.num_eval_samples,
        "mean_w1": float(w1_values.mean()),
        "std_w1": float(w1_values.std(ddof=0)),
        "mean_mmd": float(mmd_values.mean()),
        "std_mmd": float(mmd_values.std(ddof=0)),
        "mean_elapsed_seconds": float(elapsed_values.mean()),
        "std_elapsed_seconds": float(elapsed_values.std(ddof=0)),
        "mean_source_mean_assignment_radius": float(source_mean_radius_values.mean()),
        "std_source_mean_assignment_radius": float(source_mean_radius_values.std(ddof=0)),
        "mean_source_median_assignment_radius": float(source_median_radius_values.mean()),
        "std_source_median_assignment_radius": float(source_median_radius_values.std(ddof=0)),
        "mean_target_mean_assignment_radius": float(target_mean_radius_values.mean()),
        "std_target_mean_assignment_radius": float(target_mean_radius_values.std(ddof=0)),
        "mean_target_median_assignment_radius": float(target_median_radius_values.mean()),
        "std_target_median_assignment_radius": float(target_median_radius_values.std(ddof=0)),
        "mean_mean_assignment_radius": float(mean_radius_values.mean()),
        "std_mean_assignment_radius": float(mean_radius_values.std(ddof=0)),
        "mean_median_assignment_radius": float(median_radius_values.mean()),
        "std_median_assignment_radius": float(median_radius_values.std(ddof=0)),
        "quality_curve_metric": QUALITY_CURVE_METRIC,
        "quality_curve": summarize_quality_curve(seed_results),
        "seed_results": seed_results,
    }


def main() -> None:
    args = parse_args()
    anchor_counts = [int(k) for k in args.anchor_counts]
    if any(k <= 0 for k in anchor_counts):
        raise ValueError("All anchor counts must be positive.")

    device = resolve_device(args.device)
    spec = TOY_DATASETS[DEFAULT_DATASET_KEY]
    print(f"Using device: {device}")
    print(f"Algorithm: {ALGORITHM_LABEL}")
    print(f"Dataset: {spec.label}")
    print(f"Anchor counts: {', '.join(str(k) for k in anchor_counts)}")
    print(f"Sigma: {args.sigma}")
    print(f"OT method: {args.ot_method}")
    print(f"Training samples per endpoint: {args.num_samples}")
    print(f"Eval samples per endpoint: {args.num_eval_samples}")
    if args.max_eval_points <= 0:
        print("Evaluating final W1/MMD on the full evaluation populations.")
    else:
        print(f"Evaluating final W1/MMD with a {args.max_eval_points}-point cap.")
    if args.quality_eval_every > 0:
        print(f"Recording untimed MMD checkpoints every {args.quality_eval_every} epochs.")
    else:
        print("Periodic MMD checkpoints disabled.")
    print("For k=1, the Gon initializer reduces to random single-anchor selection.")

    results = []
    for coverage_anchors in anchor_counts:
        print(f"\nAnchor count: {coverage_anchors}")
        result = benchmark_anchor_count(
            coverage_anchors,
            args=args,
            device=device,
        )
        results.append(result)

    print("\nQDSB anchor sensitivity summary")
    for result in results:
        print(
            f"  k={result['coverage_anchors']}: "
            f"W1={format_metric(result['mean_w1'], result['std_w1'])} | "
            f"MMD={format_metric(result['mean_mmd'], result['std_mmd'])} | "
            f"r_mean={result['mean_mean_assignment_radius']:.6f}\u00b1{result['std_mean_assignment_radius']:.6f} | "
            f"r_med={result['mean_median_assignment_radius']:.6f}\u00b1{result['std_median_assignment_radius']:.6f}"
        )

    if args.output_json is not None:
        payload = {
            "config": {
                "algorithm": ALGORITHM_LABEL,
                "algorithm_key": ALGORITHM_KEY,
                "dataset": spec.label,
                "dataset_key": DEFAULT_DATASET_KEY,
                "seeds": args.seeds,
                "anchor_counts": anchor_counts,
                "num_samples": args.num_samples,
                "num_eval_samples": args.num_eval_samples,
                "sigma": args.sigma,
                "ot_method": args.ot_method,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "quality_eval_every": args.quality_eval_every,
                "width": args.width,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "steps_per_unit": args.steps_per_unit,
                "rollout_batch_size": args.rollout_batch_size,
                "max_eval_points": args.max_eval_points,
                "w1_method": args.w1_method,
                "w1_reg": args.w1_reg,
                "device": str(device),
                "coverage_anchor_selection": args.coverage_anchor_selection,
                "coverage_anchor_gon_plus_candidates": args.coverage_anchor_gon_plus_candidates,
                "coverage_anchor_weight_temperature": args.coverage_anchor_weight_temperature,
                "coverage_anchor_refresh_epochs": args.coverage_anchor_refresh_epochs,
                "coverage_point_match_mode": args.coverage_point_match_mode,
            },
            "results": results,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved raw metrics to {args.output_json}")


if __name__ == "__main__":
    main()
