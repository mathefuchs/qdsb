from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from toy_exp.common import compute_reference_updates, sample_minibatch
from toy_exp.quality import UpdateQualityCheckpointSchedule


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
) -> torch.Tensor:
    if timesteps.ndim == 0:
        timesteps = timesteps[None]
    if timesteps.ndim == 1:
        timesteps = timesteps[:, None]
    half_dim = embedding_dim // 2
    if half_dim == 0:
        return timesteps.float()
    emb_scale = math.log(10000) / max(half_dim - 1, 1)
    emb = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb_scale
    )
    emb = timesteps.float() * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, [0, 1])
    return emb


class TimeConditionedMLP(nn.Module):
    def __init__(self, *, x_dim: int, hidden_dim: int, time_embed_dim: int) -> None:
        super().__init__()
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
        return self.out_net(torch.cat([self.x_net(x), self.time_net(temb)], dim=-1))


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
    ) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.x_dim = x_dim
        self.gammas = gammas.float()
        self.device = device
        self.mean_final = torch.zeros(x_dim, device=device)
        self.var_final = torch.ones(x_dim, device=device)
        self.time = torch.cumsum(self.gammas, dim=0).to(self.device).float()

    def make_time_grid(self) -> torch.Tensor:
        return self.time / torch.clamp(self.time[-1], min=1e-8)

    def record_init_langevin(
        self,
        init_samples: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = init_samples
        n = x.shape[0]
        time_grid = self.make_time_grid()
        steps_expanded = time_grid.view(1, self.num_steps, 1).repeat(n, 1, 1)
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
        sample: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = init_samples
        n = x.shape[0]
        time_grid = self.make_time_grid()
        steps_expanded = time_grid.view(1, self.num_steps, 1).repeat(n, 1, 1)
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
        source_points: torch.Tensor,
        target_points: torch.Tensor,
        num_batches: int,
        batch_size: int,
        langevin: LangevinVector,
        ipf_iteration: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.data = torch.zeros((num_batches, batch_size * langevin.num_steps, 2, langevin.x_dim))
        self.steps_data = torch.zeros((num_batches, batch_size * langevin.num_steps, 1))

        sample_net_was_training = sample_net.training
        sample_net.eval()
        with torch.no_grad():
            for batch_idx in range(num_batches):
                batch = sample_minibatch(
                    source_points if fb == "b" else target_points,
                    batch_size,
                ).to(device)
                if ipf_iteration == 1 and fb == "b":
                    x, out, steps = langevin.record_init_langevin(batch)
                else:
                    x, out, steps = langevin.record_langevin_seq(sample_net, batch)
                batch_data = torch.cat((x.unsqueeze(2), out.unsqueeze(2)), dim=2)
                self.data[batch_idx] = batch_data.flatten(start_dim=0, end_dim=1).cpu()
                self.steps_data[batch_idx] = steps.flatten(start_dim=0, end_dim=1).cpu()
        if sample_net_was_training:
            sample_net.train()
        self.data = self.data.flatten(start_dim=0, end_dim=1)
        self.steps_data = self.steps_data.flatten(start_dim=0, end_dim=1)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


class PairwiseDSBTrainer:
    def __init__(
        self,
        *,
        source_points: torch.Tensor,
        target_points: torch.Tensor,
        device: torch.device,
        hyperparams: DSBHyperparams,
        progress_label: str,
    ) -> None:
        self.source_points = source_points.cpu()
        self.target_points = target_points.cpu()
        self.device = device
        self.hyperparams = hyperparams
        self.progress_label = progress_label
        self.x_dim = int(source_points.shape[1])

        if hyperparams.num_steps % 2 != 0:
            raise ValueError("num_steps must be even.")
        n_half = hyperparams.num_steps // 2
        if hyperparams.gamma_space == "linspace":
            gamma_half = np.linspace(hyperparams.gamma_min, hyperparams.gamma_max, n_half)
        elif hyperparams.gamma_space == "geomspace":
            gamma_half = np.geomspace(hyperparams.gamma_min, hyperparams.gamma_max, n_half)
        else:
            raise ValueError(f"Unsupported gamma_space: {hyperparams.gamma_space}")
        gammas = np.concatenate([gamma_half, np.flip(gamma_half)]).astype(np.float32)
        self.gammas = torch.tensor(gammas, device=device)
        self.langevin = LangevinVector(
            num_steps=hyperparams.num_steps,
            x_dim=self.x_dim,
            gammas=self.gammas,
            device=device,
        )
        self.net = nn.ModuleDict(
            {
                "f": TimeConditionedMLP(
                    x_dim=self.x_dim,
                    hidden_dim=hyperparams.hidden_dim,
                    time_embed_dim=hyperparams.time_embed_dim,
                ).to(device),
                "b": TimeConditionedMLP(
                    x_dim=self.x_dim,
                    hidden_dim=hyperparams.hidden_dim,
                    time_embed_dim=hyperparams.time_embed_dim,
                ).to(device),
            }
        )
        self.optimizer = {
            "f": optim.Adam(self.net["f"].parameters(), lr=hyperparams.lr),
            "b": optim.Adam(self.net["b"].parameters(), lr=hyperparams.lr),
        }

    def new_cacheloader(self, forward_or_backward: str, ipf_iteration: int) -> DataLoader:
        sample_direction = "f" if forward_or_backward == "b" else "b"
        cache_ds = CacheLoaderVector(
            fb=forward_or_backward,
            sample_net=self.net[sample_direction],
            source_points=self.source_points,
            target_points=self.target_points,
            num_batches=self.hyperparams.num_cache_batches,
            batch_size=self.hyperparams.cache_batch_size,
            langevin=self.langevin,
            ipf_iteration=ipf_iteration,
            device=self.device,
        )
        return DataLoader(
            cache_ds,
            batch_size=self.hyperparams.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.hyperparams.num_workers,
        )

    def ipf_step(
        self,
        forward_or_backward: str,
        ipf_iteration: int,
        *,
        progress: tqdm,
        checkpoint_schedule: UpdateQualityCheckpointSchedule | None = None,
        checkpoint_callback: Callable[[int, "PairwiseDSBTrainer"], None] | None = None,
    ) -> None:
        cache_loader = iter(self.new_cacheloader(forward_or_backward, ipf_iteration))
        optimizer = self.optimizer[forward_or_backward]
        net = self.net[forward_or_backward]
        net.train()
        running_loss = None
        for _ in range(self.hyperparams.num_iter):
            try:
                x, out, step_times = next(cache_loader)
            except StopIteration:
                cache_loader = iter(self.new_cacheloader(forward_or_backward, ipf_iteration))
                x, out, step_times = next(cache_loader)
            x = x.to(self.device)
            out = out.to(self.device)
            step_times = step_times.to(self.device)
            pred = net(x, step_times)
            loss = F.mse_loss(pred, out)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            self.completed_optimizer_updates += 1
            if checkpoint_schedule is not None and checkpoint_callback is not None:
                checkpoint_epoch = checkpoint_schedule.observe_update(self.completed_optimizer_updates)
                if checkpoint_epoch is not None:
                    checkpoint_callback(checkpoint_epoch, self)

            loss_value = float(loss.item())
            running_loss = loss_value if running_loss is None else 0.95 * running_loss + 0.05 * loss_value

        progress.update(1)
        progress.set_postfix(phase=f"{ipf_iteration}:{forward_or_backward}", loss=f"{(running_loss or 0.0):.5f}")

    def train(
        self,
        *,
        checkpoint_schedule: UpdateQualityCheckpointSchedule | None = None,
        checkpoint_callback: Callable[[int, "PairwiseDSBTrainer"], None] | None = None,
    ) -> None:
        total_steps = 2 * self.hyperparams.n_ipf
        self.completed_optimizer_updates = 0
        progress = tqdm(total=total_steps, desc=self.progress_label, leave=False, dynamic_ncols=True)
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

    @torch.inference_mode()
    def sample_forward_population(
        self,
        source_points: torch.Tensor,
        *,
        rollout_batch_size: int,
    ) -> torch.Tensor:
        outputs = []
        was_training = self.net["f"].training
        self.net["f"].eval()
        for start in range(0, source_points.shape[0], rollout_batch_size):
            batch = source_points[start: start + rollout_batch_size].to(self.device)
            x_tot, _, _ = self.langevin.record_langevin_seq(self.net["f"], batch, sample=True)
            outputs.append(x_tot[:, -1].cpu())
        if was_training:
            self.net["f"].train()
        return torch.cat(outputs, dim=0)


def resolve_effective_num_iter(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    *,
    hyperparams: DSBHyperparams,
    sf2m_epochs: int,
    manual_num_iter: int | None,
) -> tuple[int, int, int]:
    sf2m_total_updates, sf2m_steps_per_epoch = compute_reference_updates(
        source_points,
        target_points,
        batch_size=hyperparams.batch_size,
        epochs=sf2m_epochs,
    )
    if manual_num_iter is not None:
        return manual_num_iter, sf2m_steps_per_epoch, 2 * hyperparams.n_ipf * manual_num_iter
    num_iter = max(1, math.ceil(sf2m_total_updates / (2 * hyperparams.n_ipf)))
    return num_iter, sf2m_steps_per_epoch, 2 * hyperparams.n_ipf * num_iter


def train_pairwise_dsb(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    *,
    hyperparams: DSBHyperparams,
    sf2m_epochs: int,
    manual_num_iter: int | None,
    device: torch.device,
    progress_label: str,
    quality_eval_every: int,
    checkpoint_callback: Callable[[int, PairwiseDSBTrainer], None] | None = None,
) -> PairwiseDSBTrainer:
    effective_num_iter, steps_per_epoch, total_updates = resolve_effective_num_iter(
        source_points,
        target_points,
        hyperparams=hyperparams,
        sf2m_epochs=sf2m_epochs,
        manual_num_iter=manual_num_iter,
    )
    trainer = PairwiseDSBTrainer(
        source_points=source_points,
        target_points=target_points,
        device=device,
        hyperparams=replace(hyperparams, num_iter=effective_num_iter),
        progress_label=progress_label,
    )
    schedule = None
    if checkpoint_callback is not None and quality_eval_every > 0:
        schedule = UpdateQualityCheckpointSchedule(
            steps_per_epoch=steps_per_epoch,
            eval_every_epochs=quality_eval_every,
            total_updates=total_updates,
        )
    trainer.train(checkpoint_schedule=schedule, checkpoint_callback=checkpoint_callback)
    return trainer


def sample_predicted_target(
    trainer: PairwiseDSBTrainer,
    source_points: torch.Tensor,
    *,
    rollout_batch_size: int,
) -> torch.Tensor:
    return trainer.sample_forward_population(source_points, rollout_batch_size=rollout_batch_size)
