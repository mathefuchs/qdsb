from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from toy_exp.common import format_metric, resolve_device, set_seed
from toy_exp.data import TOY_DATASETS, TOY_DATASET_ORDER, sample_toy_problem
from toy_exp.dsb import DSBHyperparams, sample_predicted_target as sample_dsb_target
from toy_exp.dsb import train_pairwise_dsb
from toy_exp.dsbm import DSBMHyperparams, sample_predicted_target as sample_dsbm_target
from toy_exp.dsbm import train_pairwise_dsbm
from toy_exp.lightsb_m import DEFAULT_MIN_COV_SCALE, sample_predicted_target as sample_lightsb_target
from toy_exp.lightsb_m import train_pairwise_lightsb_m
from toy_exp.quality import (
    DEFAULT_QUALITY_CURVE_EVAL_POINTS,
    DEFAULT_QUALITY_EVAL_EVERY,
    QUALITY_CURVE_METRIC,
    QualityCheckpointRecorder,
    compute_mmd,
    compute_w1,
    estimate_mmd_gamma,
    summarize_quality_curve,
)
from toy_exp.sf2m import sample_predicted_target as sample_sf2m_target
from toy_exp.sf2m import (
    train_pairwise_qdsb,
    train_pairwise_sf2m,
    train_pairwise_sf2m_mpot,
)
from toy_exp.trajectory_net import sample_predicted_target as sample_trajectory_target
from toy_exp.trajectory_net import train_pairwise_trajectory_net

DEFAULT_ALGORITHM_ORDER = (
    # "trajectory_net",
    "sf2m",
    "sf2m_mpot",
    "qdsb",
    "dsb",
    "dsbm",
    "lightsb_m",
)
ALGORITHM_LABELS = {
    "trajectory_net": "TrajectoryNet",
    "sf2m": "SF2M",
    "sf2m_mpot": "SF2M + mPOT",
    "qdsb": "QDSB (ours)",
    "dsb": "DSB",
    "dsbm": "DSBM",
    "lightsb_m": "LightSB-M",
}
DEFAULT_COVERAGE_ANCHORS = 256
DEFAULT_COVERAGE_ANCHOR_REFRESH_EPOCHS = 100
DEFAULT_MPOT_FRACTION = 0.8
DEFAULT_TRAJECTORY_ENERGY_WEIGHT = 1e-2
DEFAULT_LIGHTSB_M_NUM_COMPONENTS = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a suite of source-target 2D toy benchmarks across multiple "
            "transport/bridge baselines."
        ),
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=list(DEFAULT_ALGORITHM_ORDER),
        choices=list(DEFAULT_ALGORITHM_ORDER),
        help="Which algorithms to run.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(TOY_DATASET_ORDER),
        choices=list(TOY_DATASET_ORDER),
        help="Which toy source-target settings to run.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--num-samples", type=int, default=16384)
    parser.add_argument("--num-eval-samples", type=int, default=4096)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--ot-method", type=str, default="exact", choices=["exact", "sinkhorn"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument(
        "--quality-eval-every",
        type=int,
        default=DEFAULT_QUALITY_EVAL_EVERY,
        help="Record elapsed time and MMD every N epochs. Use 0 to disable.",
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
    parser.add_argument("--w1-method", type=str, default="exact", choices=["exact", "sinkhorn"])
    parser.add_argument("--w1-reg", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-json", type=Path, default=None)

    parser.add_argument("--mpot-fraction", type=float, default=DEFAULT_MPOT_FRACTION)

    parser.add_argument("--coverage-anchors", type=int, default=DEFAULT_COVERAGE_ANCHORS)
    parser.add_argument(
        "--coverage-anchor-selection",
        type=str,
        default="gon",
        choices=["gon", "gon_plus"],
    )
    parser.add_argument("--coverage-anchor-gon-plus-candidates", type=int, default=5)
    parser.add_argument("--coverage-anchor-weight-temperature", type=float, default=1.0)
    parser.add_argument(
        "--coverage-anchor-refresh-epochs",
        type=int,
        default=DEFAULT_COVERAGE_ANCHOR_REFRESH_EPOCHS,
    )
    parser.add_argument(
        "--coverage-point-match-mode",
        type=str,
        default="random",
        choices=["random", "ot"],
    )

    parser.add_argument("--trajectory-energy-weight", type=float, default=DEFAULT_TRAJECTORY_ENERGY_WEIGHT)

    parser.add_argument("--dsb-num-iter", type=int, default=None)
    parser.add_argument("--dsb-n-ipf", type=int, default=20)
    parser.add_argument("--dsb-num-steps", type=int, default=20)
    parser.add_argument("--dsb-gamma-min", type=float, default=0.01)
    parser.add_argument("--dsb-gamma-max", type=float, default=0.01)
    parser.add_argument("--dsb-gamma-space", type=str, default="linspace", choices=["linspace", "geomspace"])
    parser.add_argument("--dsb-cache-batch-size", type=int, default=512)
    parser.add_argument("--dsb-num-cache-batches", type=int, default=4)
    parser.add_argument("--dsb-time-embed-dim", type=int, default=32)
    parser.add_argument("--dsb-num-workers", type=int, default=0)

    parser.add_argument("--dsbm-num-iter", type=int, default=None)
    parser.add_argument("--dsbm-n-outer", type=int, default=20)
    parser.add_argument("--dsbm-cache-batch-size", type=int, default=512)
    parser.add_argument("--dsbm-num-cache-batches", type=int, default=4)
    parser.add_argument("--dsbm-time-embed-dim", type=int, default=32)
    parser.add_argument("--dsbm-num-workers", type=int, default=0)
    parser.add_argument("--dsbm-loss-weighting", action="store_true", default=True)
    parser.add_argument("--no-dsbm-loss-weighting", dest="dsbm_loss_weighting", action="store_false")

    parser.add_argument("--lightsb-m-num-components", type=int, default=DEFAULT_LIGHTSB_M_NUM_COMPONENTS)
    parser.add_argument(
        "--lightsb-m-input-plan",
        type=str,
        default="ot",
        choices=["ot", "independent"],
    )
    parser.add_argument("--lightsb-m-min-cov-scale", type=float, default=DEFAULT_MIN_COV_SCALE)
    return parser.parse_args()


def run_single_seed(
    algorithm_key: str,
    dataset_key: str,
    *,
    train_source: torch.Tensor,
    train_target: torch.Tensor,
    eval_source: torch.Tensor,
    eval_target: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    quality_mmd_gamma = estimate_mmd_gamma(eval_target)
    checkpoint_recorder = QualityCheckpointRecorder()

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
    if algorithm_key == "sf2m":

        def callback(epoch: int, flow_model, score_model) -> None:
            checkpoint_mmd(
                sample_sf2m_target(
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

        flow_model, score_model = train_pairwise_sf2m(
            train_source,
            train_target,
            sigma=args.sigma,
            ot_method=args.ot_method,
            batch_size=args.batch_size,
            epochs=args.epochs,
            width=args.width,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            progress_label=f"{ALGORITHM_LABELS[algorithm_key]} {TOY_DATASETS[dataset_key].label} seed={seed}",
            quality_eval_every=args.quality_eval_every,
            checkpoint_callback=callback if args.quality_eval_every > 0 else None,
        )
        elapsed_seconds = checkpoint_recorder.elapsed_seconds()
        predicted = sample_sf2m_target(
            flow_model,
            score_model,
            eval_source,
            sigma=args.sigma,
            steps_per_unit=args.steps_per_unit,
            rollout_batch_size=args.rollout_batch_size,
            device=device,
        )
    elif algorithm_key == "sf2m_mpot":

        def callback(epoch: int, flow_model, score_model) -> None:
            checkpoint_mmd(
                sample_sf2m_target(
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

        flow_model, score_model = train_pairwise_sf2m_mpot(
            train_source,
            train_target,
            sigma=args.sigma,
            ot_method=args.ot_method,
            mpot_fraction=args.mpot_fraction,
            batch_size=args.batch_size,
            epochs=args.epochs,
            width=args.width,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            progress_label=f"{ALGORITHM_LABELS[algorithm_key]} {TOY_DATASETS[dataset_key].label} seed={seed}",
            quality_eval_every=args.quality_eval_every,
            checkpoint_callback=callback if args.quality_eval_every > 0 else None,
        )
        elapsed_seconds = checkpoint_recorder.elapsed_seconds()
        predicted = sample_sf2m_target(
            flow_model,
            score_model,
            eval_source,
            sigma=args.sigma,
            steps_per_unit=args.steps_per_unit,
            rollout_batch_size=args.rollout_batch_size,
            device=device,
        )
    elif algorithm_key == "qdsb":

        def callback(epoch: int, flow_model, score_model) -> None:
            checkpoint_mmd(
                sample_sf2m_target(
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

        flow_model, score_model = train_pairwise_qdsb(
            train_source,
            train_target,
            coverage_anchors=args.coverage_anchors,
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
            progress_label=f"{ALGORITHM_LABELS[algorithm_key]} {TOY_DATASETS[dataset_key].label} seed={seed}",
            quality_eval_every=args.quality_eval_every,
            checkpoint_callback=callback if args.quality_eval_every > 0 else None,
        )
        elapsed_seconds = checkpoint_recorder.elapsed_seconds()
        predicted = sample_sf2m_target(
            flow_model,
            score_model,
            eval_source,
            sigma=args.sigma,
            steps_per_unit=args.steps_per_unit,
            rollout_batch_size=args.rollout_batch_size,
            device=device,
        )
    elif algorithm_key == "trajectory_net":

        def callback(epoch: int, vector_field) -> None:
            checkpoint_mmd(
                sample_trajectory_target(
                    vector_field,
                    eval_source,
                    steps_per_unit=args.steps_per_unit,
                    rollout_batch_size=args.rollout_batch_size,
                    device=device,
                ),
                epoch=epoch,
            )

        vector_field = train_pairwise_trajectory_net(
            train_source,
            train_target,
            ot_method=args.ot_method,
            batch_size=args.batch_size,
            epochs=args.epochs,
            width=args.width,
            lr=args.lr,
            weight_decay=args.weight_decay,
            steps_per_unit=args.steps_per_unit,
            energy_weight=args.trajectory_energy_weight,
            device=device,
            progress_label=f"{ALGORITHM_LABELS[algorithm_key]} {TOY_DATASETS[dataset_key].label} seed={seed}",
            quality_eval_every=args.quality_eval_every,
            checkpoint_callback=callback if args.quality_eval_every > 0 else None,
        )
        elapsed_seconds = checkpoint_recorder.elapsed_seconds()
        predicted = sample_trajectory_target(
            vector_field,
            eval_source,
            steps_per_unit=args.steps_per_unit,
            rollout_batch_size=args.rollout_batch_size,
            device=device,
        )
    elif algorithm_key == "dsb":
        hyperparams = DSBHyperparams(
            batch_size=args.batch_size,
            cache_batch_size=args.dsb_cache_batch_size,
            num_cache_batches=args.dsb_num_cache_batches,
            num_iter=1 if args.dsb_num_iter is None else args.dsb_num_iter,
            n_ipf=args.dsb_n_ipf,
            lr=args.lr,
            num_steps=args.dsb_num_steps,
            gamma_min=args.dsb_gamma_min,
            gamma_max=args.dsb_gamma_max,
            gamma_space=args.dsb_gamma_space,
            hidden_dim=max(args.width, 128),
            time_embed_dim=args.dsb_time_embed_dim,
            num_workers=args.dsb_num_workers,
        )

        def callback(epoch: int, trainer) -> None:
            checkpoint_mmd(
                sample_dsb_target(
                    trainer,
                    eval_source,
                    rollout_batch_size=args.rollout_batch_size,
                ),
                epoch=epoch,
            )

        trainer = train_pairwise_dsb(
            train_source,
            train_target,
            hyperparams=hyperparams,
            sf2m_epochs=args.epochs,
            manual_num_iter=args.dsb_num_iter,
            device=device,
            progress_label=f"{ALGORITHM_LABELS[algorithm_key]} {TOY_DATASETS[dataset_key].label} seed={seed}",
            quality_eval_every=args.quality_eval_every,
            checkpoint_callback=callback if args.quality_eval_every > 0 else None,
        )
        elapsed_seconds = checkpoint_recorder.elapsed_seconds()
        predicted = sample_dsb_target(
            trainer,
            eval_source,
            rollout_batch_size=args.rollout_batch_size,
        )
    elif algorithm_key == "dsbm":
        hyperparams = DSBMHyperparams(
            batch_size=args.batch_size,
            cache_batch_size=args.dsbm_cache_batch_size,
            num_cache_batches=args.dsbm_num_cache_batches,
            num_iter=1 if args.dsbm_num_iter is None else args.dsbm_num_iter,
            n_outer=args.dsbm_n_outer,
            lr=args.lr,
            sigma=args.sigma,
            steps_per_unit=args.steps_per_unit,
            hidden_dim=max(args.width, 128),
            time_embed_dim=args.dsbm_time_embed_dim,
            num_workers=args.dsbm_num_workers,
            loss_weighting=args.dsbm_loss_weighting,
        )

        def callback(epoch: int, trainer) -> None:
            checkpoint_mmd(
                sample_dsbm_target(
                    trainer,
                    eval_source,
                    rollout_batch_size=args.rollout_batch_size,
                ),
                epoch=epoch,
            )

        trainer = train_pairwise_dsbm(
            train_source,
            train_target,
            hyperparams=hyperparams,
            sf2m_epochs=args.epochs,
            manual_num_iter=args.dsbm_num_iter,
            device=device,
            progress_label=f"{ALGORITHM_LABELS[algorithm_key]} {TOY_DATASETS[dataset_key].label} seed={seed}",
            quality_eval_every=args.quality_eval_every,
            checkpoint_callback=callback if args.quality_eval_every > 0 else None,
        )
        elapsed_seconds = checkpoint_recorder.elapsed_seconds()
        predicted = sample_dsbm_target(
            trainer,
            eval_source,
            rollout_batch_size=args.rollout_batch_size,
        )
    elif algorithm_key == "lightsb_m":

        def callback(epoch: int, model) -> None:
            checkpoint_mmd(
                sample_lightsb_target(
                    model,
                    eval_source,
                    sigma=args.sigma,
                    rollout_batch_size=args.rollout_batch_size,
                    device=device,
                ),
                epoch=epoch,
            )

        model = train_pairwise_lightsb_m(
            train_source,
            train_target,
            sigma=args.sigma,
            input_plan=args.lightsb_m_input_plan,
            ot_method=args.ot_method,
            batch_size=args.batch_size,
            epochs=args.epochs,
            num_components=args.lightsb_m_num_components,
            lr=args.lr,
            weight_decay=args.weight_decay,
            min_cov_scale=args.lightsb_m_min_cov_scale,
            device=device,
            progress_label=f"{ALGORITHM_LABELS[algorithm_key]} {TOY_DATASETS[dataset_key].label} seed={seed}",
            quality_eval_every=args.quality_eval_every,
            checkpoint_callback=callback if args.quality_eval_every > 0 else None,
        )
        elapsed_seconds = checkpoint_recorder.elapsed_seconds()
        predicted = sample_lightsb_target(
            model,
            eval_source,
            sigma=args.sigma,
            rollout_batch_size=args.rollout_batch_size,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm_key}")

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
    return {
        "seed": seed,
        "elapsed_seconds": float(elapsed_seconds),
        "w1": float(w1),
        "mmd": float(mmd),
        "quality_curve": checkpoint_recorder.checkpoints,
    }


def benchmark_algorithm_dataset(
    algorithm_key: str,
    dataset_key: str,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    spec = TOY_DATASETS[dataset_key]
    seed_results = []
    for seed in args.seeds:
        set_seed(seed)
        samples = sample_toy_problem(
            spec,
            num_samples=args.num_samples,
            num_eval_samples=args.num_eval_samples,
        )
        seed_result = run_single_seed(
            algorithm_key,
            dataset_key,
            train_source=samples["train_source"],
            train_target=samples["train_target"],
            eval_source=samples["eval_source"],
            eval_target=samples["eval_target"],
            args=args,
            device=device,
            seed=seed,
        )
        seed_results.append(seed_result)
        print(
            f"    seed {seed}: "
            f"W1={seed_result['w1']:.6f} | "
            f"MMD={seed_result['mmd']:.6f}"
        )

    w1_values = np.asarray([entry["w1"] for entry in seed_results], dtype=np.float64)
    mmd_values = np.asarray([entry["mmd"] for entry in seed_results], dtype=np.float64)
    elapsed_values = np.asarray([entry["elapsed_seconds"] for entry in seed_results], dtype=np.float64)
    return {
        "algorithm": ALGORITHM_LABELS[algorithm_key],
        "algorithm_key": algorithm_key,
        "dataset": spec.label,
        "dataset_key": dataset_key,
        "source": spec.source_name,
        "target": spec.target_name,
        "num_samples": args.num_samples,
        "num_eval_samples": args.num_eval_samples,
        "mean_w1": float(w1_values.mean()),
        "std_w1": float(w1_values.std(ddof=0)),
        "mean_mmd": float(mmd_values.mean()),
        "std_mmd": float(mmd_values.std(ddof=0)),
        "mean_elapsed_seconds": float(elapsed_values.mean()),
        "std_elapsed_seconds": float(elapsed_values.std(ddof=0)),
        "quality_curve_metric": QUALITY_CURVE_METRIC,
        "quality_curve": summarize_quality_curve(seed_results),
        "seed_results": seed_results,
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    print(f"Algorithms: {', '.join(ALGORITHM_LABELS[key] for key in args.algorithms)}")
    print(f"Datasets: {', '.join(TOY_DATASETS[key].label for key in args.datasets)}")
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

    results = []
    for algorithm_key in args.algorithms:
        print(f"\nAlgorithm: {ALGORITHM_LABELS[algorithm_key]}")
        for dataset_key in args.datasets:
            print(f"  Dataset: {TOY_DATASETS[dataset_key].label}")
            result = benchmark_algorithm_dataset(
                algorithm_key,
                dataset_key,
                args=args,
                device=device,
            )
            results.append(result)

    print("\nToy 2D summary")
    for algorithm_key in args.algorithms:
        print(f"  {ALGORITHM_LABELS[algorithm_key]}")
        for result in results:
            if result["algorithm_key"] != algorithm_key:
                continue
            print(
                f"    {result['dataset']} | "
                f"W1={format_metric(result['mean_w1'], result['std_w1'])} | "
                f"MMD={format_metric(result['mean_mmd'], result['std_mmd'])}"
            )

    if args.output_json is not None:
        payload = {
            "config": {
                "algorithms": args.algorithms,
                "datasets": args.datasets,
                "seeds": args.seeds,
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
                "mpot_fraction": args.mpot_fraction,
                "coverage_anchors": args.coverage_anchors,
                "coverage_anchor_selection": args.coverage_anchor_selection,
                "coverage_anchor_gon_plus_candidates": args.coverage_anchor_gon_plus_candidates,
                "coverage_anchor_weight_temperature": args.coverage_anchor_weight_temperature,
                "coverage_anchor_refresh_epochs": args.coverage_anchor_refresh_epochs,
                "coverage_point_match_mode": args.coverage_point_match_mode,
                "trajectory_energy_weight": args.trajectory_energy_weight,
                "dsb_num_iter": args.dsb_num_iter,
                "dsb_n_ipf": args.dsb_n_ipf,
                "dsb_num_steps": args.dsb_num_steps,
                "dsb_gamma_min": args.dsb_gamma_min,
                "dsb_gamma_max": args.dsb_gamma_max,
                "dsb_gamma_space": args.dsb_gamma_space,
                "dsb_cache_batch_size": args.dsb_cache_batch_size,
                "dsb_num_cache_batches": args.dsb_num_cache_batches,
                "dsb_time_embed_dim": args.dsb_time_embed_dim,
                "dsb_num_workers": args.dsb_num_workers,
                "dsbm_num_iter": args.dsbm_num_iter,
                "dsbm_n_outer": args.dsbm_n_outer,
                "dsbm_cache_batch_size": args.dsbm_cache_batch_size,
                "dsbm_num_cache_batches": args.dsbm_num_cache_batches,
                "dsbm_time_embed_dim": args.dsbm_time_embed_dim,
                "dsbm_num_workers": args.dsbm_num_workers,
                "dsbm_loss_weighting": args.dsbm_loss_weighting,
                "lightsb_m_num_components": args.lightsb_m_num_components,
                "lightsb_m_input_plan": args.lightsb_m_input_plan,
                "lightsb_m_min_cov_scale": args.lightsb_m_min_cov_scale,
            },
            "results": results,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved raw metrics to {args.output_json}")


if __name__ == "__main__":
    main()
