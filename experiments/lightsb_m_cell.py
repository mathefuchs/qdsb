"""Table-4-style single-cell interpolation with LightSB-M.

This mirrors the dataset loading and leave-one-timepoint-out evaluation
protocol from ``experiments/sf2m_cell.py`` while replacing the training loop
with the LightSB-M bridge-matching objective from
``Light and Optimal Schrödinger Bridge Matching`` (ICML 2024).

LightSB-M is a two-endpoint Schrödinger bridge solver. For each left-out
intermediate timepoint, this script trains a bridge on the bracketing observed
timepoints and evaluates the learned bridge at the missing relative time.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cell_exp.common import (build_model_times, find_bracketing_interval,
                             format_metric, resolve_dataset_sigma,
                             resolve_device, sample_minibatch, set_seed)
from cell_exp.data import (DEFAULT_DONOR, DEFAULT_PCA_EMBED_DIM,
                           TABLE4_DATASETS, TABLE4_ORDER, Table4DatasetSpec,
                           load_real_dataset)
from cell_exp.quality import (DEFAULT_QUALITY_CURVE_EVAL_POINTS,
                              DEFAULT_QUALITY_EVAL_EVERY, QUALITY_CURVE_METRIC,
                              QualityCheckpointRecorder, compute_mmd,
                              estimate_mmd_gamma, summarize_quality_curve)
from torchcfm.optimal_transport import OTPlanSampler, wasserstein
from tqdm import tqdm

EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

DEFAULT_NUM_COMPONENTS = 512
DEFAULT_MIN_COV_SCALE = 1e-2
DEFAULT_TIME_EPS = 1e-4


def inverse_softplus(value: float) -> float:
    if value <= 0:
        raise ValueError("softplus inverse expects a positive value.")
    return float(math.log(math.expm1(value)))


@dataclass(frozen=True)
class BracketingProblem:
    missing_index: int
    left_index: int
    right_index: int
    missing_relative_time: float
    interval_duration: float
    source_points: torch.Tensor
    target_points: torch.Tensor


class GaussianMixtureAdjustedPotential(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_components: int,
        init_points: torch.Tensor,
        min_cov_scale: float,
    ) -> None:
        super().__init__()
        if num_components <= 0:
            raise ValueError("num_components must be positive.")
        if init_points.ndim != 2 or init_points.shape[1] != dim:
            raise ValueError("init_points must have shape [n_points, dim].")
        if init_points.shape[0] == 0:
            raise ValueError("init_points must not be empty.")
        if min_cov_scale <= 0:
            raise ValueError("min_cov_scale must be positive.")

        self.dim = dim
        self.num_components = num_components
        self.min_cov_scale = min_cov_scale

        indices = torch.randint(
            0,
            init_points.shape[0],
            (num_components,),
        )
        means_init = init_points[indices].clone()
        var = init_points.var(dim=0, unbiased=False)
        avg_std = float(torch.sqrt(var.mean().clamp_min(1e-6)).item())
        diag_init = max(avg_std, min_cov_scale)
        raw_diag_init = inverse_softplus(diag_init - min_cov_scale)

        raw_tril = torch.zeros(num_components, dim, dim, dtype=init_points.dtype)
        diag_indices = torch.arange(dim)
        raw_tril[:, diag_indices, diag_indices] = raw_diag_init

        self.logits = nn.Parameter(torch.zeros(num_components, dtype=init_points.dtype))
        self.means = nn.Parameter(means_init)
        self.raw_tril = nn.Parameter(raw_tril)

    def scale_tril(self) -> torch.Tensor:
        lower = torch.tril(self.raw_tril, diagonal=-1)
        diag = F.softplus(torch.diagonal(self.raw_tril, dim1=-2, dim2=-1))
        diag = diag + self.min_cov_scale
        return lower + torch.diag_embed(diag)

    def component_stats(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scale_tril = self.scale_tril()
        covariance = scale_tril @ scale_tril.transpose(-1, -2)
        precision = torch.cholesky_inverse(scale_tril)
        logdet_cov = 2.0 * torch.log(
            torch.diagonal(scale_tril, dim1=-2, dim2=-1)
        ).sum(dim=-1)
        precision_means = torch.einsum("kij,kj->ki", precision, self.means)
        mean_precision_quad = torch.einsum("ki,ki->k", self.means, precision_means)
        return covariance, scale_tril, precision, logdet_cov, mean_precision_quad

    def endpoint_distribution(
        self,
        source_points: torch.Tensor,
        *,
        epsilon: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")

        covariance, scale_tril, _, _, _ = self.component_stats()
        log_mix = F.log_softmax(self.logits, dim=0)
        component_means = self.means.unsqueeze(0) + torch.einsum(
            "kij,bj->bki",
            covariance,
            source_points,
        )
        source_mean_term = source_points @ self.means.T
        source_cov_term = torch.einsum(
            "bi,kij,bj->bk",
            source_points,
            covariance,
            source_points,
        )
        log_weights = (
            log_mix.unsqueeze(0)
            + (source_mean_term + 0.5 * source_cov_term) / epsilon
        )
        component_probs = torch.softmax(log_weights, dim=1)
        component_scale_tril = math.sqrt(epsilon) * scale_tril
        return component_probs, component_means, component_scale_tril

    def drift(
        self,
        points: torch.Tensor,
        times: torch.Tensor,
        *,
        epsilon: float,
    ) -> torch.Tensor:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")
        if points.ndim != 2:
            raise ValueError("points must have shape [batch, dim].")
        if times.ndim != 1 or times.shape[0] != points.shape[0]:
            raise ValueError("times must have shape [batch].")

        covariance, _, precision, logdet_cov, mean_precision_quad = self.component_stats()
        log_mix = F.log_softmax(self.logits, dim=0)

        one_minus_t = (1.0 - times).clamp_min(DEFAULT_TIME_EPS)
        ratio = times / one_minus_t
        eye = torch.eye(self.dim, dtype=points.dtype, device=points.device)
        system = precision.unsqueeze(0) + ratio[:, None, None, None] * eye
        chol = torch.linalg.cholesky(system)

        precision_means = torch.einsum("kij,kj->ki", precision, self.means)
        rhs = points[:, None, :] / one_minus_t[:, None, None] + precision_means.unsqueeze(0)
        component_means = torch.cholesky_solve(rhs.unsqueeze(-1), chol).squeeze(-1)
        logdet_system = 2.0 * torch.log(
            torch.diagonal(chol, dim1=-2, dim2=-1)
        ).sum(dim=-1)
        rhs_quad = torch.einsum("bki,bki->bk", rhs, component_means)

        log_weights = (
            log_mix.unsqueeze(0)
            - 0.5 * logdet_cov.unsqueeze(0)
            - 0.5 * logdet_system
            + 0.5 * (rhs_quad - mean_precision_quad.unsqueeze(0)) / epsilon
        )
        component_probs = torch.softmax(log_weights, dim=1)
        endpoint_mean = torch.einsum("bk,bki->bi", component_probs, component_means)
        return (endpoint_mean - points) / one_minus_t[:, None]

    def sample_endpoint_given_source(
        self,
        source_points: torch.Tensor,
        *,
        epsilon: float,
    ) -> torch.Tensor:
        component_probs, component_means, component_scale_tril = self.endpoint_distribution(
            source_points,
            epsilon=epsilon,
        )
        component_indices = torch.multinomial(component_probs, num_samples=1).squeeze(1)
        batch_indices = torch.arange(source_points.shape[0], device=source_points.device)
        means = component_means[batch_indices, component_indices]
        chols = component_scale_tril[component_indices]
        noise = torch.randn_like(source_points)
        return means + torch.einsum("bij,bj->bi", chols, noise)


def build_bracketing_problem(
    timepoints: list[torch.Tensor],
    model_times: np.ndarray,
    *,
    missing_index: int,
) -> BracketingProblem:
    observed_indices = [idx for idx in range(len(timepoints)) if idx != missing_index]
    left_index, right_index = find_bracketing_interval(observed_indices, missing_index)
    start_time = float(model_times[left_index])
    end_time = float(model_times[right_index])
    missing_time = float(model_times[missing_index])
    interval_duration = end_time - start_time
    if interval_duration <= 0:
        raise ValueError("Model times must be strictly increasing.")
    relative_time = (missing_time - start_time) / interval_duration
    if not (0.0 < relative_time < 1.0):
        raise ValueError("Missing timepoint must lie strictly inside its bracketing interval.")
    return BracketingProblem(
        missing_index=missing_index,
        left_index=left_index,
        right_index=right_index,
        missing_relative_time=relative_time,
        interval_duration=interval_duration,
        source_points=timepoints[left_index],
        target_points=timepoints[right_index],
    )


def sample_training_plan(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    *,
    batch_size: int,
    input_plan: str,
    ot_sampler: OTPlanSampler | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    x0 = sample_minibatch(source_points, batch_size)
    x1 = sample_minibatch(target_points, batch_size)
    if input_plan == "ot":
        if ot_sampler is None:
            raise ValueError("OT input plan requires an initialized OT sampler.")
        x0, x1 = ot_sampler.sample_plan(x0, x1)
    elif input_plan != "independent":
        raise ValueError(f"Unsupported input plan: {input_plan}")
    return x0.to(device), x1.to(device)


def sample_brownian_bridge(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    times: torch.Tensor,
    *,
    sigma: float,
) -> torch.Tensor:
    bridge_mean = (
        (1.0 - times)[:, None] * source_points
        + times[:, None] * target_points
    )
    if sigma <= 0:
        return bridge_mean
    bridge_std = sigma * torch.sqrt(times * (1.0 - times))
    return bridge_mean + bridge_std[:, None] * torch.randn_like(source_points)


def train_leave_one_out_lightsb_m(
    problem: BracketingProblem,
    *,
    sigma: float,
    input_plan: str,
    ot_method: str,
    batch_size: int,
    epochs: int,
    num_components: int,
    lr: float,
    weight_decay: float,
    min_cov_scale: float,
    device: torch.device,
    progress_label: str,
    quality_eval_every: int,
    checkpoint_callback: Callable[[int, GaussianMixtureAdjustedPotential], None] | None = None,
) -> GaussianMixtureAdjustedPotential:
    if sigma <= 0:
        raise ValueError("LightSB-M requires sigma > 0.")

    dim = int(problem.source_points.shape[1])
    epsilon = sigma * sigma * problem.interval_duration
    sigma_interval = math.sqrt(epsilon)
    target_init = problem.target_points.to(dtype=torch.float32, device="cpu")
    model = GaussianMixtureAdjustedPotential(
        dim=dim,
        num_components=num_components,
        init_points=target_init,
        min_cov_scale=min_cov_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    ot_sampler = None
    if input_plan == "ot":
        sinkhorn_reg = 2.0 * epsilon
        ot_sampler = OTPlanSampler(method=ot_method, reg=sinkhorn_reg)

    max_points = max(problem.source_points.shape[0], problem.target_points.shape[0])
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
        model.train()

        for _ in range(steps_per_epoch):
            x0, x1 = sample_training_plan(
                problem.source_points,
                problem.target_points,
                batch_size=batch_size,
                input_plan=input_plan,
                ot_sampler=ot_sampler,
                device=device,
            )
            times = torch.rand(batch_size, device=device, dtype=x0.dtype)
            times = times.clamp_(min=DEFAULT_TIME_EPS, max=1.0 - DEFAULT_TIME_EPS)
            xt = sample_brownian_bridge(
                x0,
                x1,
                times,
                sigma=sigma_interval,
            )
            target_drift = (x1 - xt) / (1.0 - times)[:, None]
            predicted_drift = model.drift(xt, times, epsilon=epsilon)
            loss = torch.mean((predicted_drift - target_drift) ** 2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
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
            checkpoint_callback(epoch_number, model)

    return model


def sample_missing_population(
    model: GaussianMixtureAdjustedPotential,
    source_points: torch.Tensor,
    *,
    relative_time: float,
    epsilon: float,
    rollout_batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if not (0.0 < relative_time < 1.0):
        raise ValueError("relative_time must lie strictly between 0 and 1.")

    sigma_interval = math.sqrt(epsilon)
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, source_points.shape[0], rollout_batch_size):
            batch = source_points[start: start + rollout_batch_size].to(device)
            sampled_endpoint = model.sample_endpoint_given_source(batch, epsilon=epsilon)
            tau = torch.full(
                (batch.shape[0],),
                relative_time,
                dtype=batch.dtype,
                device=device,
            )
            xt = sample_brownian_bridge(
                batch,
                sampled_endpoint,
                tau,
                sigma=sigma_interval,
            )
            outputs.append(xt.cpu())
    return torch.cat(outputs, dim=0)


def evaluate_leave_one_out_w1(
    model: GaussianMixtureAdjustedPotential,
    problem: BracketingProblem,
    timepoints: list[torch.Tensor],
    *,
    sigma: float,
    rollout_batch_size: int,
    device: torch.device,
    max_eval_points: int | None,
    w1_method: str,
    w1_reg: float,
) -> float:
    epsilon = sigma * sigma * problem.interval_duration
    predicted = sample_missing_population(
        model,
        problem.source_points,
        relative_time=problem.missing_relative_time,
        epsilon=epsilon,
        rollout_batch_size=rollout_batch_size,
        device=device,
    )
    ground_truth = timepoints[problem.missing_index]
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
    model: GaussianMixtureAdjustedPotential,
    problem: BracketingProblem,
    timepoints: list[torch.Tensor],
    *,
    sigma: float,
    rollout_batch_size: int,
    device: torch.device,
    gamma: float,
    max_eval_points: int | None,
) -> float:
    epsilon = sigma * sigma * problem.interval_duration
    predicted = sample_missing_population(
        model,
        problem.source_points,
        relative_time=problem.missing_relative_time,
        epsilon=epsilon,
        rollout_batch_size=rollout_batch_size,
        device=device,
    )
    return compute_mmd(
        predicted,
        timepoints[problem.missing_index],
        gamma=gamma,
        max_points=max_eval_points,
    )


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
    sigma: float,
    input_plan: str,
    ot_method: str,
    batch_size: int,
    epochs: int,
    num_components: int,
    lr: float,
    weight_decay: float,
    min_cov_scale: float,
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
            problem = build_bracketing_problem(
                timepoints,
                model_times,
                missing_index=missing_index,
            )
            quality_mmd_gamma = estimate_mmd_gamma(timepoints[missing_index])
            checkpoint_recorder = QualityCheckpointRecorder()

            def checkpoint_callback(
                epoch_number: int,
                current_model: GaussianMixtureAdjustedPotential,
            ) -> None:
                checkpoint_recorder.time_evaluation(
                    epoch=epoch_number,
                    evaluate=lambda: evaluate_leave_one_out_mmd(
                        current_model,
                        problem,
                        timepoints,
                        sigma=sigma,
                        rollout_batch_size=rollout_batch_size,
                        device=device,
                        gamma=quality_mmd_gamma,
                        max_eval_points=DEFAULT_QUALITY_CURVE_EVAL_POINTS,
                    ),
                )

            model = train_leave_one_out_lightsb_m(
                problem,
                sigma=sigma,
                input_plan=input_plan,
                ot_method=ot_method,
                batch_size=batch_size,
                epochs=epochs,
                num_components=num_components,
                lr=lr,
                weight_decay=weight_decay,
                min_cov_scale=min_cov_scale,
                device=device,
                progress_label=f"{spec.label} seed={seed} miss={missing_index}",
                quality_eval_every=quality_eval_every,
                checkpoint_callback=checkpoint_callback,
            )

            w1 = evaluate_leave_one_out_w1(
                model,
                problem,
                timepoints,
                sigma=sigma,
                rollout_batch_size=rollout_batch_size,
                device=device,
                max_eval_points=max_eval_points,
                w1_method=w1_method,
                w1_reg=w1_reg,
            )
            mmd = evaluate_leave_one_out_mmd(
                model,
                problem,
                timepoints,
                sigma=sigma,
                rollout_batch_size=rollout_batch_size,
                device=device,
                gamma=quality_mmd_gamma,
                max_eval_points=max_eval_points,
            )
            leave_out_metrics.append(
                {
                    "missing_index": missing_index,
                    "left_index": problem.left_index,
                    "right_index": problem.right_index,
                    "interval_duration": problem.interval_duration,
                    "relative_time": problem.missing_relative_time,
                    "epsilon": sigma * sigma * problem.interval_duration,
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
        description="Run the Table-4-style single-cell benchmark with LightSB-M.",
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
        "--sigma",
        type=float,
        default=None,
        help=(
            "Global sigma override used for all datasets. "
            "If unset, dataset-specific defaults are used."
        ),
    )
    parser.add_argument(
        "--sigma-eb",
        type=float,
        default=0.25,
        help="Default sigma used for the EB dataset when --sigma is unset.",
    )
    parser.add_argument(
        "--sigma-cite",
        type=float,
        default=0.25,
        help="Default sigma used for the Cite dataset when --sigma is unset.",
    )
    parser.add_argument(
        "--sigma-multi",
        type=float,
        default=1.0,
        help="Default sigma used for the Multi dataset when --sigma is unset.",
    )
    parser.add_argument(
        "--time-mode",
        type=str,
        default="discrete",
        choices=["discrete", "raw", "scaled"],
        help=(
            "Time coordinates used to normalize the bracketing interval. "
            "'discrete' matches the paper's per-timepoint setup."
        ),
    )
    parser.add_argument(
        "--input-plan",
        type=str,
        default="ot",
        choices=["ot", "independent"],
        help=(
            "Plan used to sample training endpoint pairs. "
            "'ot' uses minibatch OT, 'independent' samples endpoints independently."
        ),
    )
    parser.add_argument(
        "--ot-method",
        type=str,
        default="exact",
        choices=["exact", "sinkhorn"],
        help="OT solver used when --input-plan=ot.",
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
    parser.add_argument(
        "--num-components",
        type=int,
        default=DEFAULT_NUM_COMPONENTS,
        help="Number of Gaussian components in the adjusted Schrödinger potential.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--min-cov-scale",
        type=float,
        default=DEFAULT_MIN_COV_SCALE,
        help="Minimum positive covariance scale enforced in the Gaussian mixture.",
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
    if args.sigma is None:
        print(
            "Using dataset-specific sigmas: "
            f"EB={args.sigma_eb}, Cite={args.sigma_cite}, Multi={args.sigma_multi}."
        )
    else:
        print(f"Using global sigma override: {args.sigma}.")
    if args.fit_pca:
        print(
            f"Fitting raw-data PCA embeddings with {max(args.dims, args.pca_embed_dim)} components."
        )
    print(
        "Using LightSB-M with "
        f"{args.num_components} Gaussian components and input plan {args.input_plan}."
    )
    if args.input_plan == "ot":
        print(f"Using minibatch OT method: {args.ot_method}.")
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
        dataset_sigma = resolve_dataset_sigma(
            dataset_key,
            sigma=args.sigma,
            sigma_eb=args.sigma_eb,
            sigma_cite=args.sigma_cite,
            sigma_multi=args.sigma_multi,
        )
        print(f"\nDataset: {spec.label} (sigma={dataset_sigma})")
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
            sigma=dataset_sigma,
            input_plan=args.input_plan,
            ot_method=args.ot_method,
            batch_size=args.batch_size,
            epochs=args.epochs,
            num_components=args.num_components,
            lr=args.lr,
            weight_decay=args.weight_decay,
            min_cov_scale=args.min_cov_scale,
            rollout_batch_size=args.rollout_batch_size,
            device=device,
            max_eval_points=args.max_eval_points,
            quality_eval_every=args.quality_eval_every,
            w1_method=args.w1_method,
            w1_reg=args.w1_reg,
            time_mode=args.time_mode,
        )
        result["sigma"] = dataset_sigma
        results.append(result)

    print("\nLightSB-M summary (Table 4 style)")
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
                "sigma": args.sigma,
                "sigma_eb": args.sigma_eb,
                "sigma_cite": args.sigma_cite,
                "sigma_multi": args.sigma_multi,
                "time_mode": args.time_mode,
                "input_plan": args.input_plan,
                "ot_method": args.ot_method,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "quality_eval_every": args.quality_eval_every,
                "num_components": args.num_components,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "min_cov_scale": args.min_cov_scale,
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
