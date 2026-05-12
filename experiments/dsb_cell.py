"""Table-4-style single-cell interpolation with Diffusion Schroedinger Bridges.

This mirrors the dataset loading and leave-one-timepoint-out evaluation protocol
from ``experiments/sf2m_cell.py`` while replacing the model/training loop with a
vector-valued DSB implementation adapted from ``experiments/2d_toy.py``.
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
from torch.utils.data import DataLoader, Dataset, TensorDataset
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
        self.x_dim = x_dim
        self.time_embed_dim = time_embed_dim
        self.time_net = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.x_net = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, x_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = get_timestep_embedding(t, self.time_embed_dim)
        th = self.time_net(temb)
        xh = self.x_net(x)
        return self.out_net(torch.cat([xh, th], dim=-1))


def grad_gauss(x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    return -((x - mean) / torch.clamp(var, min=1e-6))


class LangevinVector(nn.Module):
    def __init__(
        self,
        *,
        num_steps: int,
        x_dim: int,
        gammas: torch.Tensor,
        device: torch.device,
        mean_final: torch.Tensor,
        var_final: torch.Tensor,
    ) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.x_dim = x_dim
        self.gammas = gammas.float()
        self.device = device
        self.mean_final = mean_final
        self.var_final = var_final
        self.time = torch.cumsum(self.gammas, dim=0).to(self.device).float()

    def make_time_grid(
        self,
        *,
        start_time: float,
        end_time: float,
    ) -> torch.Tensor:
        if end_time <= start_time:
            raise ValueError("end_time must be larger than start_time.")
        total_duration = torch.clamp(self.time[-1], min=1e-8)
        scale = (end_time - start_time) / total_duration
        return start_time + self.time * scale

    def record_init_langevin(
        self,
        init_samples: torch.Tensor,
        *,
        time_grid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = init_samples
        n = x.shape[0]
        if time_grid.shape[0] != self.num_steps:
            raise ValueError("time_grid must match the configured num_steps.")
        steps_expanded = time_grid.to(x.device).view(1, self.num_steps, 1).repeat(
            n,
            1,
            1,
        )
        x_tot = torch.zeros(n, self.num_steps, self.x_dim, device=x.device)
        out = torch.zeros(n, self.num_steps, self.x_dim, device=x.device)

        for k in range(self.num_steps):
            gamma = self.gammas[k]
            gradx = grad_gauss(x, self.mean_final, self.var_final)
            t_old = x + gamma * gradx
            x = t_old + torch.sqrt(2.0 * gamma) * torch.randn_like(x)

            gradx_new = grad_gauss(x, self.mean_final, self.var_final)
            t_new = x + gamma * gradx_new

            x_tot[:, k] = x
            out[:, k] = t_old - t_new

        return x_tot, out, steps_expanded

    def record_langevin_seq(
        self,
        net: nn.Module,
        init_samples: torch.Tensor,
        *,
        time_grid: torch.Tensor,
        sample: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = init_samples
        n = x.shape[0]
        if time_grid.shape[0] != self.num_steps:
            raise ValueError("time_grid must match the configured num_steps.")
        steps_expanded = time_grid.to(x.device).view(1, self.num_steps, 1).repeat(
            n,
            1,
            1,
        )
        x_tot = torch.zeros(n, self.num_steps, self.x_dim, device=x.device)
        out = torch.zeros(n, self.num_steps, self.x_dim, device=x.device)

        for k in range(self.num_steps):
            gamma = self.gammas[k]
            t_k = steps_expanded[:, k]
            net_out = net(x, t_k)
            t_old = x + net_out

            if sample and k == self.num_steps - 1:
                x = t_old
            else:
                x = t_old + torch.sqrt(2.0 * gamma) * torch.randn_like(x)

            t_new = x + net(x, t_k)
            x_tot[:, k] = x
            out[:, k] = t_old - t_new

        return x_tot, out, steps_expanded


class CacheLoaderVector(Dataset):
    def __init__(
        self,
        *,
        fb: str,
        sample_net: nn.Module,
        source_dataloader,
        target_dataloader,
        num_batches: int,
        langevin: LangevinVector,
        ipf_iteration: int,
        start_time: float,
        end_time: float,
        batch_size: int,
        device: torch.device,
        show_progress: bool = False,
    ) -> None:
        super().__init__()
        self.data = torch.zeros(
            (num_batches, batch_size * langevin.num_steps, 2, langevin.x_dim)
        )
        self.steps_data = torch.zeros(
            (num_batches, batch_size * langevin.num_steps, 1))

        sample_net_was_training = sample_net.training
        sample_net.eval()
        time_grid = langevin.make_time_grid(
            start_time=start_time, end_time=end_time)

        with torch.no_grad():
            for batch_idx in tqdm(
                range(num_batches),
                desc=f"Caching {fb}",
                leave=False,
                disable=not show_progress,
            ):
                if fb == "b":
                    if source_dataloader is None:
                        raise ValueError(
                            "Backward cache requires source dataloader.")
                    batch = next(source_dataloader)[0].to(device)
                else:
                    if target_dataloader is None:
                        raise ValueError(
                            "Forward cache requires target dataloader."
                        )
                    batch = next(target_dataloader)[0].to(device)

                if ipf_iteration == 1 and fb == "b":
                    x, out, steps = langevin.record_init_langevin(
                        batch,
                        time_grid=time_grid,
                    )
                else:
                    x, out, steps = langevin.record_langevin_seq(
                        sample_net,
                        batch,
                        time_grid=time_grid,
                    )

                batch_data = torch.cat(
                    (x.unsqueeze(2), out.unsqueeze(2)), dim=2)
                self.data[batch_idx] = (
                    batch_data.flatten(start_dim=0, end_dim=1).cpu()
                )
                self.steps_data[batch_idx] = (
                    steps.flatten(start_dim=0, end_dim=1).cpu()
                )

        if sample_net_was_training:
            sample_net.train()

        self.data = self.data.flatten(start_dim=0, end_dim=1)
        self.steps_data = self.steps_data.flatten(start_dim=0, end_dim=1)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.data[index]
        return item[0], item[1], self.steps_data[index]

    def __len__(self) -> int:
        return self.data.shape[0]


@dataclass(frozen=True)
class DSBHyperparams:
    batch_size: int
    cache_batch_size: int
    num_cache_batches: int
    num_iter: int
    n_ipf: int
    lr: float
    num_steps: int
    gamma_min: float
    gamma_max: float
    gamma_space: str
    hidden_dim: int
    time_embed_dim: int
    num_workers: int


@dataclass(frozen=True)
class ObservedInterval:
    source_index: int
    target_index: int
    source_points: torch.Tensor
    target_points: torch.Tensor
    start_time: float
    end_time: float


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
    hyperparams: DSBHyperparams,
    sf2m_epochs: int,
    manual_num_iter: int | None,
) -> tuple[int, dict[str, int | None | bool]]:
    if manual_num_iter is not None:
        num_iter = manual_num_iter
        total_updates = 2 * hyperparams.n_ipf * num_iter
        return num_iter, {
            "budget_matched_to_sf2m": False,
            "sf2m_reference_epochs": sf2m_epochs,
            "sf2m_reference_steps_per_epoch": None,
            "sf2m_reference_total_updates": None,
            "dsb_total_updates": total_updates,
        }

    sf2m_total_updates, sf2m_steps_per_epoch = compute_sf2m_reference_updates(
        timepoints,
        missing_index=missing_index,
        batch_size=hyperparams.batch_size,
        epochs=sf2m_epochs,
    )
    num_directions = 2 * hyperparams.n_ipf
    num_iter = max(1, math.ceil(sf2m_total_updates / num_directions))
    dsb_total_updates = num_directions * num_iter
    return num_iter, {
        "budget_matched_to_sf2m": True,
        "sf2m_reference_epochs": sf2m_epochs,
        "sf2m_reference_steps_per_epoch": sf2m_steps_per_epoch,
        "sf2m_reference_total_updates": sf2m_total_updates,
        "dsb_total_updates": dsb_total_updates,
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


class DSBBridgeTrainer:
    def __init__(
        self,
        *,
        intervals: list[ObservedInterval],
        device: torch.device,
        hyperparams: DSBHyperparams,
        progress_label: str,
    ) -> None:
        if not intervals:
            raise ValueError(
                "DSB training requires at least one observed interval.")
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
        for interval in self.intervals:
            if interval.source_points.shape[1] != self.x_dim:
                raise ValueError(
                    "All source intervals must share the same dimension.")
            if interval.target_points.shape[1] != self.x_dim:
                raise ValueError(
                    "All target intervals must share the same dimension.")

        if hyperparams.num_steps % 2 != 0:
            raise ValueError("num_steps must be even.")

        n_half = hyperparams.num_steps // 2
        if hyperparams.gamma_space == "linspace":
            gamma_half = np.linspace(
                hyperparams.gamma_min,
                hyperparams.gamma_max,
                n_half,
            )
        elif hyperparams.gamma_space == "geomspace":
            gamma_half = np.geomspace(
                hyperparams.gamma_min,
                hyperparams.gamma_max,
                n_half,
            )
        else:
            raise ValueError(
                f"Unsupported gamma_space: {hyperparams.gamma_space}"
            )
        gammas = np.concatenate(
            [gamma_half, np.flip(gamma_half)]).astype(np.float32)
        self.gammas = torch.tensor(gammas, device=self.device)

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

        self.mean_final = torch.zeros(self.x_dim, device=self.device)
        self.var_final = torch.ones(self.x_dim, device=self.device)

        self.langevin = LangevinVector(
            num_steps=hyperparams.num_steps,
            x_dim=self.x_dim,
            gammas=self.gammas,
            device=self.device,
            mean_final=self.mean_final,
            var_final=self.var_final,
        )
        self.interval_source_loaders = []
        self.interval_target_loaders = []
        for interval in self.intervals:
            source_loader = DataLoader(
                TensorDataset(interval.source_points),
                batch_size=hyperparams.cache_batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=hyperparams.num_workers,
            )
            target_loader = DataLoader(
                TensorDataset(interval.target_points),
                batch_size=hyperparams.cache_batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=hyperparams.num_workers,
            )
            self.interval_source_loaders.append(repeater(source_loader))
            self.interval_target_loaders.append(repeater(target_loader))

    def new_cacheloader(
        self,
        forward_or_backward: str,
        ipf_iteration: int,
        *,
        show_progress: bool = False,
    ):
        sample_direction = "f" if forward_or_backward == "b" else "b"
        sample_net = self.net[sample_direction]
        cache_loaders = []
        for interval, source_loader, target_loader in zip(
            self.intervals,
            self.interval_source_loaders,
            self.interval_target_loaders,
        ):
            cache_ds = CacheLoaderVector(
                fb=forward_or_backward,
                sample_net=sample_net,
                source_dataloader=source_loader,
                target_dataloader=target_loader,
                num_batches=self.hyperparams.num_cache_batches,
                langevin=self.langevin,
                ipf_iteration=ipf_iteration,
                start_time=interval.start_time,
                end_time=interval.end_time,
                batch_size=self.hyperparams.cache_batch_size,
                device=self.device,
                show_progress=show_progress,
            )
            loader = DataLoader(
                cache_ds,
                batch_size=self.hyperparams.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=self.hyperparams.num_workers,
            )
            cache_loaders.append(repeater(loader))
        return cache_loaders

    def ipf_step(
        self,
        forward_or_backward: str,
        ipf_iteration: int,
        *,
        progress: tqdm,
        checkpoint_schedule: UpdateQualityCheckpointSchedule | None = None,
        checkpoint_callback: Callable[[
            int, "DSBBridgeTrainer"], None] | None = None,
    ) -> None:
        cache_loaders = self.new_cacheloader(
            forward_or_backward,
            ipf_iteration,
            show_progress=False,
        )
        optimizer = self.optimizer[forward_or_backward]
        net = self.net[forward_or_backward]
        net.train()

        running_loss = None
        for _ in range(self.hyperparams.num_iter):
            xs = []
            outs = []
            steps = []
            for cache_loader in cache_loaders:
                batch_x, batch_out, batch_steps = next(cache_loader)
                xs.append(batch_x)
                outs.append(batch_out)
                steps.append(batch_steps)

            x = torch.cat(xs, dim=0).to(self.device)
            out = torch.cat(outs, dim=0).to(self.device)
            step_times = torch.cat(steps, dim=0).to(self.device)

            pred = net(x, step_times)
            loss = F.mse_loss(pred, out)

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
        progress.update(1)
        progress.set_postfix(
            phase=f"{ipf_iteration}:{forward_or_backward}",
            loss=f"{(running_loss or 0.0):.5f}",
        )

    def train(
        self,
        *,
        checkpoint_schedule: UpdateQualityCheckpointSchedule | None = None,
        checkpoint_callback: Callable[[
            int, "DSBBridgeTrainer"], None] | None = None,
    ) -> None:
        total_steps = 2 * self.hyperparams.n_ipf
        self.completed_optimizer_updates = 0
        progress = tqdm(
            total=total_steps,
            desc=self.progress_label,
            leave=False,
            dynamic_ncols=True,
        )
        try:
            for ipf_iteration in range(1, self.hyperparams.n_ipf + 1):
                self.ipf_step(
                    "b",
                    ipf_iteration,
                    progress=progress,
                    checkpoint_schedule=checkpoint_schedule,
                    checkpoint_callback=checkpoint_callback,
                )
                self.ipf_step(
                    "f",
                    ipf_iteration,
                    progress=progress,
                    checkpoint_schedule=checkpoint_schedule,
                    checkpoint_callback=checkpoint_callback,
                )
        finally:
            progress.close()

    @torch.no_grad()
    def sample_forward_population(
        self,
        source_points: torch.Tensor,
        *,
        start_time: float,
        end_time: float,
        rollout_batch_size: int,
    ) -> torch.Tensor:
        if end_time <= start_time:
            return source_points.clone()

        was_training = self.net["f"].training
        self.net["f"].eval()
        time_grid = self.langevin.make_time_grid(
            start_time=start_time,
            end_time=end_time,
        )
        outputs = []
        for start in range(0, source_points.shape[0], rollout_batch_size):
            batch = source_points[start: start +
                                  rollout_batch_size].to(self.device)
            x_tot, _, _ = self.langevin.record_langevin_seq(
                self.net["f"],
                batch,
                time_grid=time_grid,
                sample=True,
            )
            outputs.append(x_tot[:, -1].cpu())
        if was_training:
            self.net["f"].train()
        return torch.cat(outputs, dim=0)


def train_leave_one_out_dsb(
    timepoints: list[torch.Tensor],
    model_times: np.ndarray,
    *,
    missing_index: int,
    hyperparams: DSBHyperparams,
    device: torch.device,
    progress_label: str,
    checkpoint_schedule: UpdateQualityCheckpointSchedule | None = None,
    checkpoint_callback: Callable[[int, DSBBridgeTrainer], None] | None = None,
) -> tuple[DSBBridgeTrainer, int, int]:
    intervals, left_idx, right_idx = build_observed_intervals(
        timepoints,
        model_times,
        missing_index=missing_index,
    )
    trainer = DSBBridgeTrainer(
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


def evaluate_leave_one_out_dsb(
    trainer: DSBBridgeTrainer,
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
    start_time = float(model_times[left_idx])
    target_time = float(model_times[missing_index])
    predicted = trainer.sample_forward_population(
        timepoints[left_idx],
        start_time=start_time,
        end_time=target_time,
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


def evaluate_leave_one_out_dsb_mmd(
    trainer: DSBBridgeTrainer,
    timepoints: list[torch.Tensor],
    model_times: np.ndarray,
    *,
    missing_index: int,
    left_idx: int,
    rollout_batch_size: int,
    gamma: float,
    max_eval_points: int | None,
) -> float:
    predicted = trainer.sample_forward_population(
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
    hyperparams: DSBHyperparams,
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
                    f"with DSB num_iter={effective_num_iter} "
                    f"(actual_updates={budget_info['dsb_total_updates']})"
                )
            else:
                print(
                    f"  missing {missing_index}: using manual DSB num_iter="
                    f"{effective_num_iter} "
                    f"(total_updates={budget_info['dsb_total_updates']})"
                )
            checkpoint_schedule = None
            if quality_eval_every > 0:
                checkpoint_schedule = UpdateQualityCheckpointSchedule(
                    steps_per_epoch=reference_steps_per_epoch,
                    eval_every_epochs=quality_eval_every,
                    total_updates=int(budget_info["dsb_total_updates"]),
                )

            def checkpoint_callback(
                epoch_number: int,
                current_trainer: DSBBridgeTrainer,
            ) -> None:
                checkpoint_recorder.time_evaluation(
                    epoch=epoch_number,
                    evaluate=lambda: evaluate_leave_one_out_dsb_mmd(
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

            trainer, left_idx, right_idx = train_leave_one_out_dsb(
                timepoints,
                model_times,
                missing_index=missing_index,
                hyperparams=effective_hyperparams,
                device=device,
                progress_label=f"{spec.label} seed={seed} miss={missing_index}",
                checkpoint_schedule=checkpoint_schedule,
                checkpoint_callback=checkpoint_callback if checkpoint_schedule is not None else None,
            )
            w1 = evaluate_leave_one_out_dsb(
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
            mmd = evaluate_leave_one_out_dsb_mmd(
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
            "Table-4-style single-cell interpolation using DSB/IPF on the "
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
            "Reference SF2M epoch budget used to match DSB optimizer updates. "
            "Ignored when --num-iter is set."
        ),
    )
    parser.add_argument(
        "--num-iter",
        type=int,
        default=None,
        help=(
            "Manual DSB optimizer updates per direction/IPF round. "
            "By default this is computed automatically to match the SF2M update "
            "budget implied by --epochs."
        ),
    )
    parser.add_argument("--n-ipf", type=int, default=20)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--gamma-min", type=float, default=0.01)
    parser.add_argument("--gamma-max", type=float, default=0.01)
    parser.add_argument(
        "--gamma-space",
        type=str,
        default="linspace",
        choices=["linspace", "geomspace"],
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--time-embed-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
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

    hyperparams = DSBHyperparams(
        batch_size=args.batch_size,
        cache_batch_size=args.cache_batch_size,
        num_cache_batches=args.num_cache_batches,
        num_iter=1 if args.num_iter is None else args.num_iter,
        n_ipf=args.n_ipf,
        lr=args.lr,
        num_steps=args.num_steps,
        gamma_min=args.gamma_min,
        gamma_max=args.gamma_max,
        gamma_space=args.gamma_space,
        hidden_dim=args.hidden_dim,
        time_embed_dim=args.time_embed_dim,
        num_workers=args.num_workers,
    )

    print(f"Using device: {device}")
    print(f"Using time mode: {args.time_mode}")
    if args.fit_pca:
        print(
            f"Fitting raw-data PCA embeddings with {max(args.dims, args.pca_embed_dim)} components."
        )
    if args.num_iter is None:
        print(
            "DSB budget matching: "
            f"matching SF2M epochs={args.epochs} with "
            f"n_ipf={args.n_ipf} and per-leave-out auto num_iter."
        )
    else:
        print(
            "DSB budget override: "
            f"using manual num_iter={args.num_iter} with n_ipf={args.n_ipf}."
        )
    print(
        "DSB hyperparameters: "
        f"num_steps={args.num_steps}, gamma=[{args.gamma_min}, {args.gamma_max}]"
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

    print("\nDSB summary (Table 4 style)")
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
                "n_ipf": args.n_ipf,
                "num_steps": args.num_steps,
                "gamma_min": args.gamma_min,
                "gamma_max": args.gamma_max,
                "gamma_space": args.gamma_space,
                "hidden_dim": args.hidden_dim,
                "time_embed_dim": args.time_embed_dim,
                "lr": args.lr,
                "num_workers": args.num_workers,
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
        args.output_json.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        print(f"\nSaved results to {args.output_json}")


if __name__ == "__main__":
    main()
