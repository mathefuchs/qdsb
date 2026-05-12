"""Table-4-style single-cell interpolation with Diffusion Schroedinger Bridge Matching.

This mirrors the dataset loading and leave-one-timepoint-out evaluation protocol
from ``experiments/sf2m_cell.py`` while replacing the training loop with the
DSBM/IMF outer-loop algorithm from ``res/dsbm.pdf`` using Brownian bridges.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, replace
from itertools import repeat
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cell_exp.common import (build_leaveout_pair_indices, build_model_times,
                             find_bracketing_interval, format_metric,
                             resolve_device, set_seed, subsample_points)
from cell_exp.data import (DEFAULT_DONOR, DEFAULT_PCA_EMBED_DIM,
                           TABLE4_DATASETS, TABLE4_ORDER, load_real_dataset)
from cell_exp.quality import (DEFAULT_QUALITY_CURVE_EVAL_POINTS,
                              DEFAULT_QUALITY_EVAL_EVERY, QUALITY_CURVE_METRIC,
                              QualityCheckpointRecorder,
                              UpdateQualityCheckpointSchedule, compute_mmd,
                              estimate_mmd_gamma, summarize_quality_curve)
from torch.utils.data import DataLoader, TensorDataset
from torchcfm.optimal_transport import wasserstein
from tqdm import tqdm

EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))


def repeater(data_loader):
    for loader in repeat(data_loader):
        for data in loader:
            yield data


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
) -> torch.Tensor:
    if timesteps.ndim == 0:
        timesteps = timesteps[None]
    if timesteps.ndim == 1:
        timesteps = timesteps[:, None]
    elif timesteps.ndim != 2 or timesteps.shape[1] != 1:
        raise ValueError(
            "timesteps must be a scalar, a 1D tensor, or a column vector."
        )

    half_dim = embedding_dim // 2
    if half_dim == 0:
        return timesteps.float()
    emb_scale = math.log(10000) / max(half_dim - 1, 1)
    emb = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        * -emb_scale
    )
    emb = timesteps.float() * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, [0, 1])
    return emb


class TimeConditionedMLP(nn.Module):
    def __init__(
        self,
        *,
        x_dim: int,
        hidden_dim: int,
        time_embed_dim: int,
    ) -> None:
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.time_net = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.x_net = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.out_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, x_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = get_timestep_embedding(t, self.time_embed_dim)
        th = self.time_net(temb)
        xh = self.x_net(x)
        return self.out_net(torch.cat([xh, th], dim=-1))


@dataclass(frozen=True)
class DSBMHyperparams:
    batch_size: int
    cache_batch_size: int
    num_cache_batches: int
    num_iter: int
    n_outer: int
    lr: float
    sigma: float
    steps_per_unit: int
    hidden_dim: int
    time_embed_dim: int
    num_workers: int
    loss_weighting: bool


@dataclass(frozen=True)
class ObservedInterval:
    source_index: int
    target_index: int
    source_points: torch.Tensor
    target_points: torch.Tensor
    start_time: float
    end_time: float


@dataclass(frozen=True)
class EndpointCoupling:
    source_points: torch.Tensor
    target_points: torch.Tensor


class EndpointBatchSource:
    def __init__(
        self,
        *,
        source_points: torch.Tensor,
        target_points: torch.Tensor,
        batch_size: int,
        num_workers: int,
        paired: bool,
    ) -> None:
        self.paired = paired
        if paired:
            loader = DataLoader(
                TensorDataset(source_points, target_points),
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=num_workers,
            )
            self.loader = repeater(loader)
            self.source_loader = None
            self.target_loader = None
        else:
            source_loader = DataLoader(
                TensorDataset(source_points),
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=num_workers,
            )
            target_loader = DataLoader(
                TensorDataset(target_points),
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=num_workers,
            )
            self.loader = None
            self.source_loader = repeater(source_loader)
            self.target_loader = repeater(target_loader)

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.paired:
            source, target = next(self.loader)
            return source, target
        source = next(self.source_loader)[0]
        target = next(self.target_loader)[0]
        if source.shape[0] == target.shape[0]:
            return source, target
        batch_size = min(source.shape[0], target.shape[0])
        if batch_size <= 0:
            raise RuntimeError("Encountered an empty endpoint batch in DSBM.")
        return source[:batch_size], target[:batch_size]


def compute_sf2m_reference_updates(
    timepoints: list[torch.Tensor],
    *,
    missing_index: int,
    batch_size: int,
    epochs: int,
) -> tuple[int, int]:
    pair_indices = build_leaveout_pair_indices(
        num_timepoints=len(timepoints),
        missing_index=missing_index,
    )
    max_points = max(
        max(timepoints[src_idx].shape[0], timepoints[dst_idx].shape[0])
        for src_idx, dst_idx in pair_indices
    )
    steps_per_epoch = max(1, math.ceil(max_points / batch_size))
    return epochs * steps_per_epoch, steps_per_epoch


def resolve_effective_num_iter(
    timepoints: list[torch.Tensor],
    *,
    missing_index: int,
    hyperparams: DSBMHyperparams,
    sf2m_epochs: int,
    manual_num_iter: int | None,
) -> tuple[int, dict[str, int | None | bool]]:
    if manual_num_iter is not None:
        num_iter = manual_num_iter
        total_updates = 2 * hyperparams.n_outer * num_iter
        return num_iter, {
            "budget_matched_to_sf2m": False,
            "sf2m_reference_epochs": sf2m_epochs,
            "sf2m_reference_steps_per_epoch": None,
            "sf2m_reference_total_updates": None,
            "dsbm_total_updates": total_updates,
        }

    sf2m_total_updates, sf2m_steps_per_epoch = compute_sf2m_reference_updates(
        timepoints,
        missing_index=missing_index,
        batch_size=hyperparams.batch_size,
        epochs=sf2m_epochs,
    )
    num_directions = 2 * hyperparams.n_outer
    num_iter = max(1, math.ceil(sf2m_total_updates / num_directions))
    dsbm_total_updates = num_directions * num_iter
    return num_iter, {
        "budget_matched_to_sf2m": True,
        "sf2m_reference_epochs": sf2m_epochs,
        "sf2m_reference_steps_per_epoch": sf2m_steps_per_epoch,
        "sf2m_reference_total_updates": sf2m_total_updates,
        "dsbm_total_updates": dsbm_total_updates,
    }


def build_observed_intervals(
    timepoints: list[torch.Tensor],
    model_times: np.ndarray,
    *,
    missing_index: int,
) -> tuple[list[ObservedInterval], int, int]:
    pair_indices = build_leaveout_pair_indices(
        num_timepoints=len(timepoints),
        missing_index=missing_index,
    )
    intervals = [
        ObservedInterval(
            source_index=src_idx,
            target_index=dst_idx,
            source_points=timepoints[src_idx],
            target_points=timepoints[dst_idx],
            start_time=float(model_times[src_idx]),
            end_time=float(model_times[dst_idx]),
        )
        for src_idx, dst_idx in pair_indices
    ]
    observed_indices = [idx for idx in range(
        len(timepoints)) if idx != missing_index]
    left_idx, right_idx = find_bracketing_interval(
        observed_indices, missing_index)
    return intervals, left_idx, right_idx


def sample_brownian_bridge_batch(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    *,
    start_time: float,
    end_time: float,
    sigma: float,
    loss_weighting: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    delta = end_time - start_time
    if delta <= 0:
        raise ValueError("Brownian bridge requires end_time > start_time.")

    tau = torch.rand((source_points.shape[0], 1), device=source_points.device)
    tau = tau.clamp_(1e-4, 1.0 - 1e-4)
    tau_flat = tau.squeeze(1)

    noise = torch.randn_like(source_points)
    bridge_std = sigma * math.sqrt(delta) * torch.sqrt(tau * (1.0 - tau))
    xt = (1.0 - tau) * source_points + tau * target_points + bridge_std * noise

    elapsed = delta * tau
    remaining = delta * (1.0 - tau)
    global_t = start_time + delta * tau_flat

    forward_target = (target_points - xt) / remaining
    backward_target = (source_points - xt) / elapsed

    if loss_weighting:
        forward_weight = 1.0 / (
            1.0 + sigma * sigma * tau_flat / (delta * (1.0 - tau_flat))
        )
        backward_weight = 1.0 / (
            1.0 + sigma * sigma * (1.0 - tau_flat) / (delta * tau_flat)
        )
    else:
        forward_weight = torch.ones_like(tau_flat)
        backward_weight = torch.ones_like(tau_flat)

    return (
        global_t,
        xt,
        forward_target,
        backward_target,
        forward_weight,
        backward_weight,
    )


class DSBMBridgeTrainer:
    def __init__(
        self,
        *,
        intervals: list[ObservedInterval],
        device: torch.device,
        hyperparams: DSBMHyperparams,
        progress_label: str,
    ) -> None:
        if not intervals:
            raise ValueError(
                "DSBM training requires at least one observed interval."
            )
        if hyperparams.sigma < 0:
            raise ValueError("sigma must be non-negative.")
        if hyperparams.steps_per_unit <= 0:
            raise ValueError("steps_per_unit must be positive.")

        self.device = device
        self.hyperparams = hyperparams
        self.progress_label = progress_label
        self.intervals = [
            ObservedInterval(
                source_index=interval.source_index,
                target_index=interval.target_index,
                source_points=interval.source_points.cpu(),
                target_points=interval.target_points.cpu(),
                start_time=interval.start_time,
                end_time=interval.end_time,
            )
            for interval in intervals
        ]
        self.x_dim = int(self.intervals[0].source_points.shape[1])
        self.net = nn.ModuleDict(
            {
                "f": TimeConditionedMLP(
                    x_dim=self.x_dim,
                    hidden_dim=hyperparams.hidden_dim,
                    time_embed_dim=hyperparams.time_embed_dim,
                ).to(self.device),
                "b": TimeConditionedMLP(
                    x_dim=self.x_dim,
                    hidden_dim=hyperparams.hidden_dim,
                    time_embed_dim=hyperparams.time_embed_dim,
                ).to(self.device),
            }
        )
        self.optimizer = {
            "f": optim.Adam(self.net["f"].parameters(), lr=hyperparams.lr),
            "b": optim.Adam(self.net["b"].parameters(), lr=hyperparams.lr),
        }
        self.current_couplings: list[EndpointCoupling] | None = None

    def _num_rollout_steps(self, start_time: float, end_time: float) -> int:
        duration = end_time - start_time
        if duration <= 0:
            return 1
        return max(1, int(math.ceil(duration * self.hyperparams.steps_per_unit)))

    def _make_training_sources(self) -> list[EndpointBatchSource]:
        if self.current_couplings is None:
            return [
                EndpointBatchSource(
                    source_points=interval.source_points,
                    target_points=interval.target_points,
                    batch_size=self.hyperparams.batch_size,
                    num_workers=self.hyperparams.num_workers,
                    paired=False,
                )
                for interval in self.intervals
            ]
        return [
            EndpointBatchSource(
                source_points=coupling.source_points,
                target_points=coupling.target_points,
                batch_size=self.hyperparams.batch_size,
                num_workers=self.hyperparams.num_workers,
                paired=True,
            )
            for coupling in self.current_couplings
        ]

    def _single_tensor_loader(
        self,
        points: torch.Tensor,
        *,
        batch_size: int,
    ):
        loader = DataLoader(
            TensorDataset(points),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.hyperparams.num_workers,
        )
        return repeater(loader)

    def _rollout_forward_sde(
        self,
        state: torch.Tensor,
        *,
        start_time: float,
        end_time: float,
    ) -> torch.Tensor:
        duration = end_time - start_time
        if duration <= 0:
            return state.clone()

        steps = self._num_rollout_steps(start_time, end_time)
        dt = duration / steps
        sqrt_dt = math.sqrt(dt)
        x = state
        net = self.net["f"]
        was_training = net.training
        net.eval()
        with torch.inference_mode():
            for step in range(steps):
                current_t = start_time + step * dt
                t = torch.full(
                    (x.shape[0],),
                    current_t,
                    dtype=x.dtype,
                    device=x.device,
                )
                drift = net(x, t)
                x = x + dt * drift
                if self.hyperparams.sigma > 0:
                    x = x + self.hyperparams.sigma * \
                        sqrt_dt * torch.randn_like(x)
        if was_training:
            net.train()
        return x

    def _rollout_backward_sde(
        self,
        state: torch.Tensor,
        *,
        start_time: float,
        end_time: float,
    ) -> torch.Tensor:
        duration = end_time - start_time
        if duration <= 0:
            return state.clone()

        steps = self._num_rollout_steps(start_time, end_time)
        dt = duration / steps
        sqrt_dt = math.sqrt(dt)
        x = state
        net = self.net["b"]
        was_training = net.training
        net.eval()
        with torch.inference_mode():
            for step in range(steps):
                current_t = end_time - step * dt
                t = torch.full(
                    (x.shape[0],),
                    current_t,
                    dtype=x.dtype,
                    device=x.device,
                )
                drift = net(x, t)
                x = x + dt * drift
                if self.hyperparams.sigma > 0:
                    x = x + self.hyperparams.sigma * \
                        sqrt_dt * torch.randn_like(x)
        if was_training:
            net.train()
        return x

    def _build_reciprocal_couplings(self, direction: str) -> list[EndpointCoupling]:
        couplings = []
        for interval in self.intervals:
            if direction == "f":
                source_loader = self._single_tensor_loader(
                    interval.source_points,
                    batch_size=self.hyperparams.cache_batch_size,
                )
                source_batches = []
                target_batches = []
                for _ in range(self.hyperparams.num_cache_batches):
                    source = next(source_loader)[0].to(self.device)
                    target = self._rollout_forward_sde(
                        source,
                        start_time=interval.start_time,
                        end_time=interval.end_time,
                    )
                    source_batches.append(source.cpu())
                    target_batches.append(target.cpu())
            elif direction == "b":
                target_loader = self._single_tensor_loader(
                    interval.target_points,
                    batch_size=self.hyperparams.cache_batch_size,
                )
                source_batches = []
                target_batches = []
                for _ in range(self.hyperparams.num_cache_batches):
                    target = next(target_loader)[0].to(self.device)
                    source = self._rollout_backward_sde(
                        target,
                        start_time=interval.start_time,
                        end_time=interval.end_time,
                    )
                    source_batches.append(source.cpu())
                    target_batches.append(target.cpu())
            else:
                raise ValueError(f"Unsupported DSBM direction: {direction}")

            couplings.append(
                EndpointCoupling(
                    source_points=torch.cat(source_batches, dim=0),
                    target_points=torch.cat(target_batches, dim=0),
                )
            )
        return couplings

    def _train_phase(
        self,
        direction: str,
        outer_iteration: int,
        *,
        progress: tqdm,
        checkpoint_schedule: UpdateQualityCheckpointSchedule | None = None,
        checkpoint_callback: Callable[[
            int, "DSBMBridgeTrainer"], None] | None = None,
    ) -> None:
        sources = self._make_training_sources()
        net = self.net[direction]
        optimizer = self.optimizer[direction]
        net.train()

        running_loss = None
        for _ in range(self.hyperparams.num_iter):
            batch_x = []
            batch_t = []
            batch_target = []
            batch_weight = []

            for interval, source in zip(self.intervals, sources):
                endpoint_source, endpoint_target = source.next()
                endpoint_source = endpoint_source.to(self.device)
                endpoint_target = endpoint_target.to(self.device)

                (
                    global_t,
                    xt,
                    forward_target,
                    backward_target,
                    forward_weight,
                    backward_weight,
                ) = sample_brownian_bridge_batch(
                    endpoint_source,
                    endpoint_target,
                    start_time=interval.start_time,
                    end_time=interval.end_time,
                    sigma=self.hyperparams.sigma,
                    loss_weighting=self.hyperparams.loss_weighting,
                )
                if direction == "f":
                    batch_target.append(forward_target)
                    batch_weight.append(forward_weight)
                else:
                    batch_target.append(backward_target)
                    batch_weight.append(backward_weight)
                batch_x.append(xt)
                batch_t.append(global_t)

            x = torch.cat(batch_x, dim=0)
            t = torch.cat(batch_t, dim=0)
            target = torch.cat(batch_target, dim=0)
            weight = torch.cat(batch_weight, dim=0)

            pred = net(x, t)
            loss = (((pred - target) ** 2).mean(dim=1) * weight).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            self.completed_optimizer_updates += 1
            if checkpoint_schedule is not None and checkpoint_callback is not None:
                checkpoint_epoch = checkpoint_schedule.observe_update(
                    self.completed_optimizer_updates
                )
                if checkpoint_epoch is not None:
                    checkpoint_callback(checkpoint_epoch, self)

            loss_value = float(loss.item())
            if running_loss is None:
                running_loss = loss_value
            else:
                running_loss = 0.95 * running_loss + 0.05 * loss_value

        self.current_couplings = self._build_reciprocal_couplings(direction)
        progress.update(1)
        progress.set_postfix(
            phase=f"{outer_iteration}:{direction}",
            loss=f"{(running_loss or 0.0):.5f}",
        )

    def train(
        self,
        *,
        checkpoint_schedule: UpdateQualityCheckpointSchedule | None = None,
        checkpoint_callback: Callable[[
            int, "DSBMBridgeTrainer"], None] | None = None,
    ) -> None:
        total_steps = 2 * self.hyperparams.n_outer
        self.completed_optimizer_updates = 0
        progress = tqdm(
            total=total_steps,
            desc=self.progress_label,
            leave=False,
            dynamic_ncols=True,
        )
        try:
            for outer_iteration in range(1, self.hyperparams.n_outer + 1):
                self._train_phase(
                    "b",
                    outer_iteration,
                    progress=progress,
                    checkpoint_schedule=checkpoint_schedule,
                    checkpoint_callback=checkpoint_callback,
                )
                self._train_phase(
                    "f",
                    outer_iteration,
                    progress=progress,
                    checkpoint_schedule=checkpoint_schedule,
                    checkpoint_callback=checkpoint_callback,
                )
        finally:
            progress.close()

    @torch.no_grad()
    def sample_probability_flow(
        self,
        source_points: torch.Tensor,
        *,
        start_time: float,
        end_time: float,
        rollout_batch_size: int,
    ) -> torch.Tensor:
        duration = end_time - start_time
        if duration <= 0:
            return source_points.clone()

        steps = self._num_rollout_steps(start_time, end_time)
        dt = duration / steps
        outputs = []
        forward_net = self.net["f"]
        backward_net = self.net["b"]
        forward_was_training = forward_net.training
        backward_was_training = backward_net.training
        forward_net.eval()
        backward_net.eval()
        try:
            with torch.inference_mode():
                for start in range(0, source_points.shape[0], rollout_batch_size):
                    x = source_points[start: start +
                                      rollout_batch_size].to(self.device)
                    for step in range(steps):
                        current_t = start_time + step * dt
                        t = torch.full(
                            (x.shape[0],),
                            current_t,
                            dtype=x.dtype,
                            device=x.device,
                        )
                        drift = 0.5 * (forward_net(x, t) - backward_net(x, t))
                        x = x + dt * drift
                    outputs.append(x.cpu())
        finally:
            if forward_was_training:
                forward_net.train()
            if backward_was_training:
                backward_net.train()
        return torch.cat(outputs, dim=0)


def train_leave_one_out_dsbm(
    timepoints: list[torch.Tensor],
    model_times: np.ndarray,
    *,
    missing_index: int,
    hyperparams: DSBMHyperparams,
    device: torch.device,
    progress_label: str,
    checkpoint_schedule: UpdateQualityCheckpointSchedule | None = None,
    checkpoint_callback: Callable[[
        int, DSBMBridgeTrainer], None] | None = None,
) -> tuple[DSBMBridgeTrainer, int, int]:
    intervals, left_idx, right_idx = build_observed_intervals(
        timepoints,
        model_times,
        missing_index=missing_index,
    )
    trainer = DSBMBridgeTrainer(
        intervals=intervals,
        device=device,
        hyperparams=hyperparams,
        progress_label=progress_label,
    )
    trainer.train(
        checkpoint_schedule=checkpoint_schedule,
        checkpoint_callback=checkpoint_callback,
    )
    return trainer, left_idx, right_idx


def evaluate_leave_one_out_dsbm(
    trainer: DSBMBridgeTrainer,
    timepoints: list[torch.Tensor],
    model_times: np.ndarray,
    *,
    missing_index: int,
    left_idx: int,
    rollout_batch_size: int,
    max_eval_points: int | None,
    w1_method: str,
    w1_reg: float,
) -> float:
    predicted = trainer.sample_probability_flow(
        timepoints[left_idx],
        start_time=float(model_times[left_idx]),
        end_time=float(model_times[missing_index]),
        rollout_batch_size=rollout_batch_size,
    )
    predicted = subsample_points(predicted, max_eval_points)
    ground_truth = subsample_points(timepoints[missing_index], max_eval_points)
    return wasserstein(
        predicted,
        ground_truth,
        method=w1_method,
        reg=w1_reg,
        power=1,
    )


def evaluate_leave_one_out_dsbm_mmd(
    trainer: DSBMBridgeTrainer,
    timepoints: list[torch.Tensor],
    model_times: np.ndarray,
    *,
    missing_index: int,
    left_idx: int,
    rollout_batch_size: int,
    gamma: float,
    max_eval_points: int | None,
) -> float:
    predicted = trainer.sample_probability_flow(
        timepoints[left_idx],
        start_time=float(model_times[left_idx]),
        end_time=float(model_times[missing_index]),
        rollout_batch_size=rollout_batch_size,
    )
    return compute_mmd(
        predicted,
        timepoints[missing_index],
        gamma=gamma,
        max_points=max_eval_points,
    )


def benchmark_dataset(
    dataset_key: str,
    *,
    data_root: Path,
    dims: int,
    pca_embed_dim: int,
    fit_pca: bool,
    whiten: bool,
    pca_batch_size: int,
    donor: int,
    seeds: list[int],
    hyperparams: DSBMHyperparams,
    sf2m_epochs: int,
    manual_num_iter: int | None,
    rollout_batch_size: int,
    device: torch.device,
    max_eval_points: int | None,
    quality_eval_every: int,
    w1_method: str,
    w1_reg: float,
    time_mode: str,
) -> dict[str, object]:
    spec = TABLE4_DATASETS[dataset_key]
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
            observed_indices = [
                idx for idx in range(len(timepoints)) if idx != missing_index
            ]
            left_idx, _ = find_bracketing_interval(
                observed_indices, missing_index)
            quality_mmd_gamma = estimate_mmd_gamma(timepoints[missing_index])
            _, reference_steps_per_epoch = compute_sf2m_reference_updates(
                timepoints,
                missing_index=missing_index,
                batch_size=hyperparams.batch_size,
                epochs=sf2m_epochs,
            )
            effective_num_iter, budget_info = resolve_effective_num_iter(
                timepoints,
                missing_index=missing_index,
                hyperparams=hyperparams,
                sf2m_epochs=sf2m_epochs,
                manual_num_iter=manual_num_iter,
            )
            effective_hyperparams = replace(
                hyperparams,
                num_iter=effective_num_iter,
            )
            if manual_num_iter is None:
                print(
                    f"  missing {missing_index}: matching SF2M budget "
                    f"(epochs={sf2m_epochs}, "
                    f"steps/epoch={budget_info['sf2m_reference_steps_per_epoch']}, "
                    f"target_updates={budget_info['sf2m_reference_total_updates']}) "
                    f"with DSBM num_iter={effective_num_iter} "
                    f"(actual_updates={budget_info['dsbm_total_updates']})"
                )
            else:
                print(
                    f"  missing {missing_index}: using manual DSBM num_iter="
                    f"{effective_num_iter} "
                    f"(total_updates={budget_info['dsbm_total_updates']})"
                )
            checkpoint_schedule = None
            if quality_eval_every > 0:
                checkpoint_schedule = UpdateQualityCheckpointSchedule(
                    steps_per_epoch=reference_steps_per_epoch,
                    eval_every_epochs=quality_eval_every,
                    total_updates=int(budget_info["dsbm_total_updates"]),
                )

            def checkpoint_callback(
                epoch_number: int,
                current_trainer: DSBMBridgeTrainer,
            ) -> None:
                checkpoint_recorder.time_evaluation(
                    epoch=epoch_number,
                    evaluate=lambda: evaluate_leave_one_out_dsbm_mmd(
                        current_trainer,
                        timepoints,
                        model_times,
                        missing_index=missing_index,
                        left_idx=left_idx,
                        rollout_batch_size=rollout_batch_size,
                        gamma=quality_mmd_gamma,
                        max_eval_points=DEFAULT_QUALITY_CURVE_EVAL_POINTS,
                    ),
                )

            trainer, left_idx, right_idx = train_leave_one_out_dsbm(
                timepoints,
                model_times,
                missing_index=missing_index,
                hyperparams=effective_hyperparams,
                device=device,
                progress_label=f"{spec.label} seed={seed} miss={missing_index}",
                checkpoint_schedule=checkpoint_schedule,
                checkpoint_callback=checkpoint_callback if checkpoint_schedule is not None else None,
            )
            w1 = evaluate_leave_one_out_dsbm(
                trainer,
                timepoints,
                model_times,
                missing_index=missing_index,
                left_idx=left_idx,
                rollout_batch_size=rollout_batch_size,
                max_eval_points=max_eval_points,
                w1_method=w1_method,
                w1_reg=w1_reg,
            )
            mmd = evaluate_leave_one_out_dsbm_mmd(
                trainer,
                timepoints,
                model_times,
                missing_index=missing_index,
                left_idx=left_idx,
                rollout_batch_size=rollout_batch_size,
                gamma=quality_mmd_gamma,
                max_eval_points=max_eval_points,
            )
            leave_out_metrics.append(
                {
                    "missing_index": missing_index,
                    "w1": float(w1),
                    "mmd": float(mmd),
                    "left_index": left_idx,
                    "right_index": right_idx,
                    "num_iter": effective_num_iter,
                    "n_outer": hyperparams.n_outer,
                    "sigma": hyperparams.sigma,
                    "quality_curve": checkpoint_recorder.checkpoints,
                    **budget_info,
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
    mmd_means = np.asarray(
        [entry["mean_mmd"] for entry in seed_results],
        dtype=np.float64,
    )
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
        description=(
            "Table-4-style single-cell interpolation using DSBM/IMF on the "
            "same retained observed intervals as the SF2M leave-one-out setup."
        ),
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
    parser.add_argument("--dims", type=int, default=5)
    parser.add_argument(
        "--pca-embed-dim",
        type=int,
        default=DEFAULT_PCA_EMBED_DIM,
        help="Number of PCA components to build before truncation.",
    )
    parser.set_defaults(fit_pca=True, loss_weighting=True)
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
    parser.add_argument("--pca-batch-size", type=int, default=512)
    parser.add_argument("--donor", type=int, default=DEFAULT_DONOR)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cache-batch-size", type=int, default=512)
    parser.add_argument("--num-cache-batches", type=int, default=4)
    parser.add_argument(
        "--epochs",
        type=int,
        default=10000,
        help=(
            "Reference SF2M epoch budget used to match DSBM optimizer updates. "
            "Ignored when --num-iter is set."
        ),
    )
    parser.add_argument(
        "--num-iter",
        type=int,
        default=None,
        help=(
            "Manual DSBM optimizer updates per direction and outer iteration. "
            "By default this is computed automatically to match the SF2M update "
            "budget implied by --epochs."
        ),
    )
    parser.add_argument("--n-outer", type=int, default=20)
    parser.add_argument("--steps-per-unit", type=int, default=20)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--time-embed-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--no-loss-weighting",
        dest="loss_weighting",
        action="store_false",
        help="Disable the Brownian-bridge loss downweighting from the DSBM paper.",
    )
    parser.add_argument(
        "--time-mode",
        type=str,
        default="discrete",
        choices=["discrete", "raw", "scaled"],
    )
    parser.add_argument("--rollout-batch-size", type=int, default=2048)
    parser.add_argument(
        "--quality-eval-every",
        type=int,
        default=DEFAULT_QUALITY_EVAL_EVERY,
        help=(
            "Record elapsed time and MMD every N matched SF2M epochs. "
            "Use 0 to disable periodic quality checkpoints."
        ),
    )
    parser.add_argument(
        "--max-eval-points",
        type=int,
        default=0,
        help="Subsample cap for the 1-Wasserstein evaluation. Use 0 for full populations.",
    )
    parser.add_argument(
        "--w1-method",
        type=str,
        default="exact",
        choices=["exact", "sinkhorn"],
    )
    parser.add_argument("--w1-reg", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    whiten = not args.no_whiten
    max_eval_points = None if args.max_eval_points <= 0 else args.max_eval_points

    hyperparams = DSBMHyperparams(
        batch_size=args.batch_size,
        cache_batch_size=args.cache_batch_size,
        num_cache_batches=args.num_cache_batches,
        num_iter=1 if args.num_iter is None else args.num_iter,
        n_outer=args.n_outer,
        lr=args.lr,
        sigma=args.sigma,
        steps_per_unit=args.steps_per_unit,
        hidden_dim=args.hidden_dim,
        time_embed_dim=args.time_embed_dim,
        num_workers=args.num_workers,
        loss_weighting=args.loss_weighting,
    )

    print(f"Using device: {device}")
    print(f"Using time mode: {args.time_mode}")
    if args.fit_pca:
        print(
            f"Fitting raw-data PCA embeddings with {max(args.dims, args.pca_embed_dim)} components."
        )
    if args.num_iter is None:
        print(
            "DSBM budget matching: "
            f"matching SF2M epochs={args.epochs} with "
            f"n_outer={args.n_outer} and per-leave-out auto num_iter."
        )
    else:
        print(
            "DSBM budget override: "
            f"using manual num_iter={args.num_iter} with n_outer={args.n_outer}."
        )
    print(
        "DSBM hyperparameters: "
        f"sigma={args.sigma}, steps_per_unit={args.steps_per_unit}, "
        f"loss_weighting={args.loss_weighting}"
    )
    if max_eval_points is None:
        print("Evaluating W1 on full pushed-forward populations.")
    else:
        print(f"Evaluating W1 with a {max_eval_points}-point cap.")
    if args.quality_eval_every > 0:
        print(
            "Recording quality checkpoints every "
            f"{args.quality_eval_every} matched SF2M epochs using MMD."
        )
    else:
        print("Periodic quality checkpoints disabled.")

    results = []
    for dataset_key in args.datasets:
        spec = TABLE4_DATASETS[dataset_key]
        print(f"\nDataset: {spec.label}")
        result = benchmark_dataset(
            dataset_key,
            data_root=args.data_root,
            dims=args.dims,
            pca_embed_dim=args.pca_embed_dim,
            fit_pca=args.fit_pca,
            whiten=whiten,
            pca_batch_size=args.pca_batch_size,
            donor=args.donor,
            seeds=args.seeds,
            hyperparams=hyperparams,
            sf2m_epochs=args.epochs,
            manual_num_iter=args.num_iter,
            rollout_batch_size=args.rollout_batch_size,
            device=device,
            max_eval_points=max_eval_points,
            quality_eval_every=args.quality_eval_every,
            w1_method=args.w1_method,
            w1_reg=args.w1_reg,
            time_mode=args.time_mode,
        )
        results.append(result)

    print("\nDSBM summary (Table 4 style)")
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
                "batch_size": args.batch_size,
                "cache_batch_size": args.cache_batch_size,
                "num_cache_batches": args.num_cache_batches,
                "epochs": args.epochs,
                "num_iter": args.num_iter,
                "n_outer": args.n_outer,
                "steps_per_unit": args.steps_per_unit,
                "sigma": args.sigma,
                "hidden_dim": args.hidden_dim,
                "time_embed_dim": args.time_embed_dim,
                "lr": args.lr,
                "num_workers": args.num_workers,
                "loss_weighting": args.loss_weighting,
                "time_mode": args.time_mode,
                "rollout_batch_size": args.rollout_batch_size,
                "quality_eval_every": args.quality_eval_every,
                "max_eval_points": args.max_eval_points,
                "w1_method": args.w1_method,
                "w1_reg": args.w1_reg,
                "device": str(device),
            },
            "results": results,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
