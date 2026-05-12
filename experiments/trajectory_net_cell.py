"""Table-4-style single-cell interpolation with a TrajectoryNet-style baseline.

This reuses the same dataset loading, leave-one-out evaluation, and MLP width
defaults as ``experiments/sf2m_cell.py``. The training algorithm is a
TrajectoryNet-inspired Neural ODE baseline using:

- a shared time-conditioned vector field across observed intervals,
- minibatch OT endpoint matching at each observed interval, and
- the dynamic-OT energy regularizer from TrajectoryNet ("Base + E").

The original TrajectoryNet paper uses CNF likelihood terms and additional
regularizers for specific biological settings. For fairness to the present
benchmark, this implementation keeps the same model size and default optimizer
settings as the other cell baselines and uses the energy-regularized transport
objective as the core adaptation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from cell_exp.common import (build_leaveout_pair_indices, build_model_times,
                             format_metric, resolve_device, sample_minibatch,
                             set_seed)
from cell_exp.data import (DEFAULT_DONOR, DEFAULT_PCA_EMBED_DIM,
                           TABLE4_DATASETS, TABLE4_ORDER, Table4DatasetSpec,
                           load_real_dataset)
from cell_exp.quality import (DEFAULT_QUALITY_CURVE_EVAL_POINTS,
                              DEFAULT_QUALITY_EVAL_EVERY, QUALITY_CURVE_METRIC,
                              QualityCheckpointRecorder, estimate_mmd_gamma,
                              summarize_quality_curve)
from cell_exp.sf2m import make_time_input
from torchcfm.models import MLP
from torchcfm.optimal_transport import OTPlanSampler, wasserstein
from tqdm import tqdm

EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

DEFAULT_ENERGY_WEIGHT = 1e-2


class TrajectoryNetVectorField(nn.Module):
    def __init__(self, *, dim: int, width: int, device: torch.device) -> None:
        super().__init__()
        self.net = MLP(dim=dim, time_varying=True, w=width).to(device)

    def velocity(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(make_time_input(x, t))

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = torch.full((x.shape[0],), float(t.item()), dtype=x.dtype, device=x.device)
        return self.velocity(x, t)


def rk4_integrate_interval(
    vector_field: TrajectoryNetVectorField,
    state: torch.Tensor,
    *,
    start_time: float,
    end_time: float,
    steps: int,
    track_energy: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    dt = (end_time - start_time) / steps
    if dt <= 0:
        raise ValueError("end_time must be larger than start_time.")

    x = state
    total_energy = x.new_zeros(())
    for step in range(steps):
        current_t = start_time + step * dt
        t0 = torch.full((x.shape[0],), current_t, dtype=x.dtype, device=x.device)
        half_t = torch.full((x.shape[0],), current_t + 0.5 * dt, dtype=x.dtype, device=x.device)
        end_t = torch.full((x.shape[0],), current_t + dt, dtype=x.dtype, device=x.device)

        k1 = vector_field.velocity(x, t0)
        k2 = vector_field.velocity(x + 0.5 * dt * k1, half_t)
        k3 = vector_field.velocity(x + 0.5 * dt * k2, half_t)
        k4 = vector_field.velocity(x + dt * k3, end_t)
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        if track_energy:
            total_energy = total_energy + dt * torch.mean(torch.sum(k1 * k1, dim=1))

    if track_energy:
        return x, total_energy
    return x, None


def rollout_to_missing_time(
    vector_field: TrajectoryNetVectorField,
    source_points: torch.Tensor,
    *,
    start_time: float,
    target_time: float,
    steps_per_unit: int,
    rollout_batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    duration = target_time - start_time
    if duration <= 0:
        raise ValueError("target_time must be larger than start_time.")
    steps = max(1, int(math.ceil(duration * steps_per_unit)))

    outputs = []
    vector_field.eval()
    with torch.inference_mode():
        for start in range(0, source_points.shape[0], rollout_batch_size):
            batch = source_points[start: start + rollout_batch_size].to(device)
            out, _ = rk4_integrate_interval(
                vector_field,
                batch,
                start_time=start_time,
                end_time=target_time,
                steps=steps,
                track_energy=False,
            )
            outputs.append(out.cpu())
    return torch.cat(outputs, dim=0)


def evaluate_leave_one_out(
    vector_field: TrajectoryNetVectorField,
    timepoints: list[torch.Tensor],
    model_times: np.ndarray,
    *,
    missing_index: int,
    steps_per_unit: int,
    rollout_batch_size: int,
    device: torch.device,
    max_eval_points: int | None,
    w1_method: str,
    w1_reg: float,
) -> float:
    observed_indices = [idx for idx in range(len(timepoints)) if idx != missing_index]
    left_idx = max(idx for idx in observed_indices if idx < missing_index)
    predicted = rollout_to_missing_time(
        vector_field,
        timepoints[left_idx],
        start_time=float(model_times[left_idx]),
        target_time=float(model_times[missing_index]),
        steps_per_unit=steps_per_unit,
        rollout_batch_size=rollout_batch_size,
        device=device,
    )
    ground_truth = timepoints[missing_index]
    if max_eval_points is not None and max_eval_points > 0:
        predicted = predicted[torch.randperm(predicted.shape[0])[:max_eval_points]]
        ground_truth = ground_truth[torch.randperm(ground_truth.shape[0])[:max_eval_points]]
    return wasserstein(
        predicted,
        ground_truth,
        method=w1_method,
        reg=w1_reg,
        power=1,
    )


def evaluate_leave_one_out_mmd(
    vector_field: TrajectoryNetVectorField,
    timepoints: list[torch.Tensor],
    model_times: np.ndarray,
    *,
    missing_index: int,
    steps_per_unit: int,
    rollout_batch_size: int,
    device: torch.device,
    gamma: float,
    max_eval_points: int | None,
) -> float:
    observed_indices = [idx for idx in range(len(timepoints)) if idx != missing_index]
    left_idx = max(idx for idx in observed_indices if idx < missing_index)
    predicted = rollout_to_missing_time(
        vector_field,
        timepoints[left_idx],
        start_time=float(model_times[left_idx]),
        target_time=float(model_times[missing_index]),
        steps_per_unit=steps_per_unit,
        rollout_batch_size=rollout_batch_size,
        device=device,
    )
    from cell_exp.quality import compute_mmd

    return compute_mmd(
        predicted,
        timepoints[missing_index],
        gamma=gamma,
        max_points=max_eval_points,
    )


def train_leave_one_out_trajectory_net(
    timepoints: list[torch.Tensor],
    model_times: np.ndarray,
    *,
    missing_index: int,
    ot_method: str,
    batch_size: int,
    epochs: int,
    width: int,
    lr: float,
    weight_decay: float,
    steps_per_unit: int,
    energy_weight: float,
    device: torch.device,
    progress_label: str,
    quality_eval_every: int,
    checkpoint_callback: Callable[[int, TrajectoryNetVectorField], None] | None = None,
) -> TrajectoryNetVectorField:
    dim = int(timepoints[0].shape[1])
    vector_field = TrajectoryNetVectorField(dim=dim, width=width, device=device)
    optimizer = torch.optim.AdamW(
        vector_field.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    ot_sampler = OTPlanSampler(method=ot_method)
    pair_indices = build_leaveout_pair_indices(
        num_timepoints=len(timepoints),
        missing_index=missing_index,
    )
    max_points = max(
        max(timepoints[src_idx].shape[0], timepoints[dst_idx].shape[0])
        for src_idx, dst_idx in pair_indices
    )

    steps_per_epoch = max(1, math.ceil(max_points / batch_size))
    progress = tqdm(
        range(epochs),
        desc=progress_label,
        leave=False,
        dynamic_ncols=True,
    )
    running_loss = None

    for epoch_idx in progress:
        epoch_loss = 0.0
        n_updates = 0
        vector_field.train()

        for _ in range(steps_per_epoch):
            total_loss = None

            for src_idx, dst_idx in pair_indices:
                start_time = float(model_times[src_idx])
                end_time = float(model_times[dst_idx])
                duration = end_time - start_time
                if duration <= 0:
                    raise ValueError("Model times must be strictly increasing.")

                x0 = sample_minibatch(timepoints[src_idx], batch_size).to(device)
                x1 = sample_minibatch(timepoints[dst_idx], batch_size).to(device)
                ode_steps = max(1, int(math.ceil(duration * steps_per_unit)))
                predicted, energy = rk4_integrate_interval(
                    vector_field,
                    x0,
                    start_time=start_time,
                    end_time=end_time,
                    steps=ode_steps,
                    track_energy=True,
                )
                plan = ot_sampler.get_map(predicted.detach(), x1.detach())
                src_plan_idx, dst_plan_idx = ot_sampler.sample_map(
                    plan,
                    predicted.shape[0],
                    replace=True,
                )
                src_plan_idx = torch.from_numpy(src_plan_idx).to(device=device, dtype=torch.long)
                dst_plan_idx = torch.from_numpy(dst_plan_idx).to(device=device, dtype=torch.long)
                matched_predicted = predicted[src_plan_idx]
                matched_target = x1[dst_plan_idx]
                endpoint_loss = torch.mean((matched_predicted - matched_target) ** 2)
                interval_loss = endpoint_loss + energy_weight * energy
                total_loss = interval_loss if total_loss is None else total_loss + interval_loss

            total_loss = total_loss / len(pair_indices)
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            epoch_loss += float(total_loss.item())
            n_updates += 1

        mean_loss = epoch_loss / max(n_updates, 1)
        if running_loss is None:
            running_loss = mean_loss
        else:
            running_loss = 0.95 * running_loss + 0.05 * mean_loss
        progress.set_postfix(loss=f"{running_loss:.5f}")

        epoch_number = epoch_idx + 1
        should_record = (
            checkpoint_callback is not None
            and quality_eval_every > 0
            and (
                epoch_number % quality_eval_every == 0
                or epoch_number == epochs
            )
        )
        if should_record:
            checkpoint_callback(epoch_number, vector_field)

    return vector_field


def benchmark_dataset(
    spec: Table4DatasetSpec,
    *,
    data_root: Path,
    dims: int,
    pca_embed_dim: int,
    fit_pca: bool,
    whiten: bool,
    pca_batch_size: int,
    donor: int,
    seeds: list[int],
    ot_method: str,
    batch_size: int,
    epochs: int,
    width: int,
    lr: float,
    weight_decay: float,
    steps_per_unit: int,
    energy_weight: float,
    rollout_batch_size: int,
    device: torch.device,
    max_eval_points: int | None,
    quality_eval_every: int,
    w1_method: str,
    w1_reg: float,
    time_mode: str,
) -> dict[str, object]:
    timepoints, raw_times, artifact_desc = load_real_dataset(
        spec,
        data_root=data_root,
        dims=dims,
        pca_embed_dim=pca_embed_dim,
        fit_pca=fit_pca,
        whiten=whiten,
        pca_batch_size=pca_batch_size,
        donor=donor,
    )
    model_times = build_model_times(raw_times, time_mode)
    seed_results = []

    for seed in seeds:
        set_seed(seed)
        leave_out_metrics = []
        for missing_index in spec.leave_out:
            checkpoint_recorder = QualityCheckpointRecorder()
            quality_mmd_gamma = estimate_mmd_gamma(timepoints[missing_index])

            def checkpoint_callback(
                epoch_number: int,
                current_vector_field: TrajectoryNetVectorField,
            ) -> None:
                checkpoint_recorder.time_evaluation(
                    epoch=epoch_number,
                    evaluate=lambda: evaluate_leave_one_out_mmd(
                        current_vector_field,
                        timepoints,
                        model_times,
                        missing_index=missing_index,
                        steps_per_unit=steps_per_unit,
                        rollout_batch_size=rollout_batch_size,
                        device=device,
                        gamma=quality_mmd_gamma,
                        max_eval_points=DEFAULT_QUALITY_CURVE_EVAL_POINTS,
                    ),
                )

            vector_field = train_leave_one_out_trajectory_net(
                timepoints,
                model_times,
                missing_index=missing_index,
                ot_method=ot_method,
                batch_size=batch_size,
                epochs=epochs,
                width=width,
                lr=lr,
                weight_decay=weight_decay,
                steps_per_unit=steps_per_unit,
                energy_weight=energy_weight,
                device=device,
                progress_label=f"{spec.label} seed={seed} miss={missing_index}",
                quality_eval_every=quality_eval_every,
                checkpoint_callback=checkpoint_callback,
            )
            w1 = evaluate_leave_one_out(
                vector_field,
                timepoints,
                model_times,
                missing_index=missing_index,
                steps_per_unit=steps_per_unit,
                rollout_batch_size=rollout_batch_size,
                device=device,
                max_eval_points=max_eval_points,
                w1_method=w1_method,
                w1_reg=w1_reg,
            )
            mmd = evaluate_leave_one_out_mmd(
                vector_field,
                timepoints,
                model_times,
                missing_index=missing_index,
                steps_per_unit=steps_per_unit,
                rollout_batch_size=rollout_batch_size,
                device=device,
                gamma=quality_mmd_gamma,
                max_eval_points=max_eval_points,
            )
            leave_out_metrics.append(
                {
                    "missing_index": missing_index,
                    "w1": float(w1),
                    "mmd": float(mmd),
                    "quality_curve": checkpoint_recorder.checkpoints,
                }
            )

        mean_w1 = float(np.mean([entry["w1"] for entry in leave_out_metrics]))
        mean_mmd = float(np.mean([entry["mmd"] for entry in leave_out_metrics]))
        seed_results.append(
            {
                "seed": seed,
                "mean_w1": mean_w1,
                "mean_mmd": mean_mmd,
                "leave_out": leave_out_metrics,
            }
        )
        print(f"  seed {seed}: W1={mean_w1:.6f} | MMD={mean_mmd:.6f}")

    w1_means = np.asarray([entry["mean_w1"] for entry in seed_results], dtype=np.float64)
    mmd_means = np.asarray([entry["mean_mmd"] for entry in seed_results], dtype=np.float64)
    return {
        "dataset": spec.label,
        "artifact": artifact_desc,
        "times": raw_times.tolist(),
        "model_times": model_times.tolist(),
        "mean_w1": float(w1_means.mean()),
        "std_w1": float(w1_means.std(ddof=0)),
        "mean_mmd": float(mmd_means.mean()),
        "std_mmd": float(mmd_means.std(ddof=0)),
        "quality_curve_metric": QUALITY_CURVE_METRIC,
        "quality_curve": summarize_quality_curve(seed_results),
        "seed_results": seed_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Table-4-style single-cell benchmark with a TrajectoryNet baseline.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["eb", "cite", "multi"],
        choices=list(TABLE4_ORDER),
        help="Which Table 4 datasets to run.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Repository data root containing cite_multi and embryoid.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Model seeds used for the paper summary.",
    )
    parser.add_argument("--dims", type=int, default=5, help="Number of feature dimensions to use.")
    parser.add_argument(
        "--pca-embed-dim",
        type=int,
        default=DEFAULT_PCA_EMBED_DIM,
        help=(
            "Number of PCA components to build before whitening/truncating. "
            "Used only when fitting PCA from the raw local data."
        ),
    )
    parser.set_defaults(fit_pca=True)
    parser.add_argument(
        "--no-pca",
        dest="fit_pca",
        action="store_false",
        help="Skip PCA and use the first --dims features directly.",
    )
    parser.add_argument(
        "--no-whiten",
        action="store_true",
        help="Disable per-dimension whitening after truncation/PCA.",
    )
    parser.add_argument(
        "--pca-batch-size",
        type=int,
        default=512,
        help="Batch size used when fitting/transfoming PCA from raw data.",
    )
    parser.add_argument(
        "--donor",
        type=int,
        default=DEFAULT_DONOR,
        help="Donor id used for Cite and Multi.",
    )
    parser.add_argument(
        "--time-mode",
        type=str,
        default="discrete",
        choices=["discrete", "raw", "scaled"],
        help=(
            "Time coordinates used by the shared trajectory model. "
            "'discrete' matches the paper's per-timepoint integration setup."
        ),
    )
    parser.add_argument(
        "--ot-method",
        type=str,
        default="exact",
        choices=["exact", "sinkhorn"],
        help="Endpoint OT solver used during training.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument(
        "--quality-eval-every",
        type=int,
        default=DEFAULT_QUALITY_EVAL_EVERY,
        help=(
            "Record elapsed time and MMD every N epochs. "
            "Use 0 to disable periodic quality checkpoints."
        ),
    )
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--steps-per-unit",
        type=int,
        default=100,
        help="Fixed RK4 steps per unit of model time.",
    )
    parser.add_argument(
        "--energy-weight",
        type=float,
        default=DEFAULT_ENERGY_WEIGHT,
        help="Weight of the TrajectoryNet path-energy regularizer.",
    )
    parser.add_argument("--rollout-batch-size", type=int, default=2048)
    parser.add_argument(
        "--max-eval-points",
        type=int,
        default=0,
        help=(
            "Subsample cap for the 1-Wasserstein evaluation. "
            "Use 0 to evaluate on the full pushed-forward populations."
        ),
    )
    parser.add_argument(
        "--w1-method",
        type=str,
        default="exact",
        choices=["exact", "sinkhorn"],
        help="Method used for the leave-one-out 1-Wasserstein metric.",
    )
    parser.add_argument(
        "--w1-reg",
        type=float,
        default=0.05,
        help="Regularization used when --w1-method=sinkhorn.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    whiten = not args.no_whiten
    print(f"Using device: {device}")
    print(f"Using time mode: {args.time_mode}")
    if args.fit_pca:
        print(
            f"Fitting raw-data PCA embeddings with {max(args.dims, args.pca_embed_dim)} components."
        )
    print(
        "Using a TrajectoryNet-style Neural ODE with "
        f"energy weight {args.energy_weight} and OT method {args.ot_method}."
    )
    if args.max_eval_points <= 0:
        print("Evaluating W1 on full pushed-forward populations.")
    else:
        print(f"Evaluating W1 with a {args.max_eval_points}-point cap.")
    if args.quality_eval_every > 0:
        print(
            "Recording quality checkpoints every "
            f"{args.quality_eval_every} epochs using MMD."
        )
    else:
        print("Periodic quality checkpoints disabled.")

    results = []
    for dataset_key in args.datasets:
        spec = TABLE4_DATASETS[dataset_key]
        print(f"\nDataset: {spec.label}")
        result = benchmark_dataset(
            spec,
            data_root=args.data_root,
            dims=args.dims,
            pca_embed_dim=args.pca_embed_dim,
            fit_pca=args.fit_pca,
            whiten=whiten,
            pca_batch_size=args.pca_batch_size,
            donor=args.donor,
            seeds=args.seeds,
            ot_method=args.ot_method,
            batch_size=args.batch_size,
            epochs=args.epochs,
            width=args.width,
            lr=args.lr,
            weight_decay=args.weight_decay,
            steps_per_unit=args.steps_per_unit,
            energy_weight=args.energy_weight,
            rollout_batch_size=args.rollout_batch_size,
            device=device,
            max_eval_points=args.max_eval_points,
            quality_eval_every=args.quality_eval_every,
            w1_method=args.w1_method,
            w1_reg=args.w1_reg,
            time_mode=args.time_mode,
        )
        results.append(result)

    print("\nTrajectoryNet summary (Table 4 style)")
    for result in results:
        print(
            f"  {result['dataset']} | "
            f"W1={format_metric(result['mean_w1'], result['std_w1'])} | "
            f"MMD={format_metric(result['mean_mmd'], result['std_mmd'])}"
        )

    if args.output_json is not None:
        payload = {
            "config": {
                "datasets": args.datasets,
                "data_root": str(args.data_root),
                "seeds": args.seeds,
                "dims": args.dims,
                "pca_embed_dim": args.pca_embed_dim,
                "fit_pca": args.fit_pca,
                "whiten": whiten,
                "pca_batch_size": args.pca_batch_size,
                "donor": args.donor,
                "time_mode": args.time_mode,
                "ot_method": args.ot_method,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "quality_eval_every": args.quality_eval_every,
                "width": args.width,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "steps_per_unit": args.steps_per_unit,
                "energy_weight": args.energy_weight,
                "rollout_batch_size": args.rollout_batch_size,
                "max_eval_points": args.max_eval_points,
                "w1_method": args.w1_method,
                "w1_reg": args.w1_reg,
                "device": str(device),
            },
            "results": results,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved results to {args.output_json}")


if __name__ == "__main__":
    main()
