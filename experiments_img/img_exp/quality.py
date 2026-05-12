from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from img_exp.common import subsample_points
from torchcfm.optimal_transport import wasserstein

QUALITY_CURVE_METRIC = "mmd"
DEFAULT_QUALITY_EVAL_EVERY = 0
DEFAULT_QUALITY_CURVE_EVAL_POINTS = 0
DEFAULT_MMD_MEDIAN_HEURISTIC_POINTS = 4096
DEFAULT_MMD_CHUNK_SIZE = 2048


def deterministic_subsample_points(
    points: torch.Tensor,
    *,
    num_points: int,
    seed: int,
) -> torch.Tensor:
    if points.shape[0] <= num_points:
        return points
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randperm(points.shape[0], generator=generator)[:num_points]
    return points[indices.to(points.device)]


def estimate_mmd_gamma(
    reference_points: torch.Tensor,
    *,
    max_points: int = DEFAULT_MMD_MEDIAN_HEURISTIC_POINTS,
) -> float:
    reference = deterministic_subsample_points(
        reference_points,
        num_points=max_points,
        seed=0,
    ).to(dtype=torch.float32, device="cpu")
    if reference.shape[0] < 2:
        return 1.0
    sq_dists = torch.pdist(reference, p=2).pow_(2)
    positive = sq_dists[sq_dists > 0]
    if positive.numel() == 0:
        return 1.0
    median_sq_dist = float(positive.median().item())
    return 1.0 / max(median_sq_dist, 1e-12)


def _rbf_kernel_sum(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    gamma: float,
    chunk_size: int,
) -> float:
    total = 0.0
    for x_start in range(0, x.shape[0], chunk_size):
        x_chunk = x[x_start: x_start + chunk_size]
        x_norm = (x_chunk**2).sum(dim=1, keepdim=True)
        for y_start in range(0, y.shape[0], chunk_size):
            y_chunk = y[y_start: y_start + chunk_size]
            y_norm = (y_chunk**2).sum(dim=1).unsqueeze(0)
            sq_dists = (x_norm + y_norm - 2.0 * x_chunk @ y_chunk.T).clamp_min_(0.0)
            total += float(torch.exp(-gamma * sq_dists).sum().item())
    return total


def compute_mmd(
    predicted: torch.Tensor,
    ground_truth: torch.Tensor,
    *,
    gamma: float,
    max_points: int | None,
    chunk_size: int = DEFAULT_MMD_CHUNK_SIZE,
) -> float:
    predicted = subsample_points(predicted, max_points).to(dtype=torch.float32, device="cpu")
    ground_truth = subsample_points(ground_truth, max_points).to(dtype=torch.float32, device="cpu")
    n = predicted.shape[0]
    m = ground_truth.shape[0]
    if n == 0 or m == 0:
        raise ValueError("MMD requires non-empty predicted and reference samples.")

    sum_xx = _rbf_kernel_sum(predicted, predicted, gamma=gamma, chunk_size=chunk_size)
    sum_yy = _rbf_kernel_sum(ground_truth, ground_truth, gamma=gamma, chunk_size=chunk_size)
    sum_xy = _rbf_kernel_sum(predicted, ground_truth, gamma=gamma, chunk_size=chunk_size)
    if n > 1 and m > 1:
        mmd2 = (
            (sum_xx - n) / (n * (n - 1))
            + (sum_yy - m) / (m * (m - 1))
            - 2.0 * sum_xy / (n * m)
        )
    else:
        mmd2 = sum_xx / (n * n) + sum_yy / (m * m) - 2.0 * sum_xy / (n * m)
    return float(np.sqrt(max(mmd2, 0.0)))


def compute_w1(
    predicted: torch.Tensor,
    ground_truth: torch.Tensor,
    *,
    max_points: int | None,
    method: str,
    reg: float,
) -> float:
    predicted = subsample_points(predicted, max_points)
    ground_truth = subsample_points(ground_truth, max_points)
    return float(
        wasserstein(
            predicted,
            ground_truth,
            method=method,
            reg=reg,
            power=1,
        )
    )


@dataclass
class QualityCheckpointRecorder:
    metric_key: str = QUALITY_CURVE_METRIC
    start_time: float = field(default_factory=time.perf_counter)
    cumulative_eval_seconds: float = 0.0
    checkpoints: list[dict[str, float | int]] = field(default_factory=list)

    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.start_time - self.cumulative_eval_seconds

    def time_block(self, fn: Callable[[], float | dict[str, float] | object]) -> float | dict[str, float] | object:
        eval_start = time.perf_counter()
        try:
            return fn()
        finally:
            self.cumulative_eval_seconds += time.perf_counter() - eval_start

    def time_evaluation(
        self,
        *,
        epoch: int,
        evaluate: Callable[[], float],
    ) -> None:
        metric_value = float(self.time_block(evaluate))
        self.checkpoints.append(
            {
                "epoch": epoch,
                "elapsed_seconds": float(self.elapsed_seconds()),
                self.metric_key: metric_value,
            }
        )


@dataclass
class UpdateQualityCheckpointSchedule:
    steps_per_epoch: int
    eval_every_epochs: int
    total_updates: int | None = None
    next_checkpoint_update: int = field(init=False)

    def __post_init__(self) -> None:
        if self.steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be positive.")
        if self.eval_every_epochs <= 0:
            raise ValueError("eval_every_epochs must be positive.")
        if self.total_updates is not None and self.total_updates <= 0:
            raise ValueError("total_updates must be positive when provided.")
        self.next_checkpoint_update = self.eval_every_epochs * self.steps_per_epoch

    def observe_update(self, completed_updates: int) -> int | None:
        if completed_updates <= 0:
            raise ValueError("completed_updates must be positive.")
        interval_due = completed_updates >= self.next_checkpoint_update
        final_due = self.total_updates is not None and completed_updates == self.total_updates
        if not interval_due and not final_due:
            return None
        while completed_updates >= self.next_checkpoint_update:
            self.next_checkpoint_update += self.eval_every_epochs * self.steps_per_epoch
        return max(1, int(np.ceil(completed_updates / self.steps_per_epoch)))


def summarize_quality_curve(
    seed_results: list[dict[str, object]],
    *,
    metric_key: str = QUALITY_CURVE_METRIC,
) -> list[dict[str, float | int]]:
    checkpoints_by_epoch: dict[int, dict[str, list[float]]] = {}
    for seed_result in seed_results:
        for checkpoint in seed_result.get("quality_curve", []):
            epoch = int(checkpoint["epoch"])
            bucket = checkpoints_by_epoch.setdefault(
                epoch,
                {"elapsed_seconds": [], metric_key: []},
            )
            bucket["elapsed_seconds"].append(float(checkpoint["elapsed_seconds"]))
            bucket[metric_key].append(float(checkpoint[metric_key]))

    summary = []
    for epoch in sorted(checkpoints_by_epoch):
        bucket = checkpoints_by_epoch[epoch]
        elapsed = np.asarray(bucket["elapsed_seconds"], dtype=np.float64)
        metric_values = np.asarray(bucket[metric_key], dtype=np.float64)
        summary.append(
            {
                "epoch": epoch,
                "mean_elapsed_seconds": float(elapsed.mean()),
                "std_elapsed_seconds": float(elapsed.std(ddof=0)),
                f"mean_{metric_key}": float(metric_values.mean()),
                f"std_{metric_key}": float(metric_values.std(ddof=0)),
                "num_measurements": int(metric_values.size),
            }
        )
    return summary
