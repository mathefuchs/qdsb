from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from toy_exp.common import compute_reference_updates, sample_minibatch
from toy_exp.dsb import TimeConditionedMLP
from toy_exp.quality import UpdateQualityCheckpointSchedule


def sample_brownian_bridge_batch(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    *,
    sigma: float,
    loss_weighting: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tau = torch.rand((source_points.shape[0], 1), device=source_points.device).clamp_(1e-4, 1.0 - 1e-4)
    tau_flat = tau.squeeze(1)
    noise = torch.randn_like(source_points)
    bridge_std = sigma * torch.sqrt(tau * (1.0 - tau))
    xt = (1.0 - tau) * source_points + tau * target_points + bridge_std * noise
    forward_target = (target_points - xt) / (1.0 - tau)
    backward_target = (source_points - xt) / tau
    if loss_weighting:
        forward_weight = 1.0 / (1.0 + sigma * sigma * tau_flat / (1.0 - tau_flat))
        backward_weight = 1.0 / (1.0 + sigma * sigma * (1.0 - tau_flat) / tau_flat)
    else:
        forward_weight = torch.ones_like(tau_flat)
        backward_weight = torch.ones_like(tau_flat)
    return tau_flat, xt, forward_target, backward_target, forward_weight, backward_weight


@dataclass(frozen=True)
class EndpointCoupling:
    source_points: torch.Tensor
    target_points: torch.Tensor


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


class PairwiseDSBMTrainer:
    def __init__(
        self,
        *,
        source_points: torch.Tensor,
        target_points: torch.Tensor,
        device: torch.device,
        hyperparams: DSBMHyperparams,
        progress_label: str,
    ) -> None:
        self.source_points = source_points.cpu()
        self.target_points = target_points.cpu()
        self.device = device
        self.hyperparams = hyperparams
        self.progress_label = progress_label
        self.x_dim = int(source_points.shape[1])
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
        self.current_coupling: EndpointCoupling | None = None

    def _num_rollout_steps(self) -> int:
        return max(1, int(math.ceil(self.hyperparams.steps_per_unit)))

    def _sample_training_endpoints(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.current_coupling is None:
            return (
                sample_minibatch(self.source_points, self.hyperparams.batch_size),
                sample_minibatch(self.target_points, self.hyperparams.batch_size),
            )
        source_batch = sample_minibatch(self.current_coupling.source_points, self.hyperparams.batch_size)
        target_batch = sample_minibatch(self.current_coupling.target_points, self.hyperparams.batch_size)
        batch_size = min(source_batch.shape[0], target_batch.shape[0])
        return source_batch[:batch_size], target_batch[:batch_size]

    @torch.inference_mode()
    def _rollout_forward_sde(self, state: torch.Tensor) -> torch.Tensor:
        steps = self._num_rollout_steps()
        dt = 1.0 / steps
        sqrt_dt = math.sqrt(dt)
        x = state
        net = self.net["f"]
        was_training = net.training
        net.eval()
        for step in range(steps):
            current_t = step * dt
            t = torch.full((x.shape[0],), current_t, dtype=x.dtype, device=x.device)
            x = x + dt * net(x, t)
            if self.hyperparams.sigma > 0:
                x = x + self.hyperparams.sigma * sqrt_dt * torch.randn_like(x)
        if was_training:
            net.train()
        return x

    @torch.inference_mode()
    def _rollout_backward_sde(self, state: torch.Tensor) -> torch.Tensor:
        steps = self._num_rollout_steps()
        dt = 1.0 / steps
        sqrt_dt = math.sqrt(dt)
        x = state
        net = self.net["b"]
        was_training = net.training
        net.eval()
        for step in range(steps):
            current_t = 1.0 - step * dt
            t = torch.full((x.shape[0],), current_t, dtype=x.dtype, device=x.device)
            x = x + dt * net(x, t)
            if self.hyperparams.sigma > 0:
                x = x + self.hyperparams.sigma * sqrt_dt * torch.randn_like(x)
        if was_training:
            net.train()
        return x

    def _build_reciprocal_coupling(self, direction: str) -> EndpointCoupling:
        source_batches = []
        target_batches = []
        for _ in range(self.hyperparams.num_cache_batches):
            if direction == "f":
                source = sample_minibatch(self.source_points, self.hyperparams.cache_batch_size).to(self.device)
                target = self._rollout_forward_sde(source)
            elif direction == "b":
                target = sample_minibatch(self.target_points, self.hyperparams.cache_batch_size).to(self.device)
                source = self._rollout_backward_sde(target)
            else:
                raise ValueError(f"Unsupported DSBM direction: {direction}")
            source_batches.append(source.cpu())
            target_batches.append(target.cpu())
        return EndpointCoupling(
            source_points=torch.cat(source_batches, dim=0),
            target_points=torch.cat(target_batches, dim=0),
        )

    def _train_phase(
        self,
        direction: str,
        outer_iteration: int,
        *,
        progress: tqdm,
        checkpoint_schedule: UpdateQualityCheckpointSchedule | None = None,
        checkpoint_callback: Callable[[int, "PairwiseDSBMTrainer"], None] | None = None,
    ) -> None:
        net = self.net[direction]
        optimizer = self.optimizer[direction]
        net.train()
        running_loss = None
        for _ in range(self.hyperparams.num_iter):
            endpoint_source, endpoint_target = self._sample_training_endpoints()
            endpoint_source = endpoint_source.to(self.device)
            endpoint_target = endpoint_target.to(self.device)
            tau, xt, forward_target, backward_target, forward_weight, backward_weight = (
                sample_brownian_bridge_batch(
                    endpoint_source,
                    endpoint_target,
                    sigma=self.hyperparams.sigma,
                    loss_weighting=self.hyperparams.loss_weighting,
                )
            )
            target = forward_target if direction == "f" else backward_target
            weight = forward_weight if direction == "f" else backward_weight
            pred = net(xt, tau)
            loss = (((pred - target) ** 2).mean(dim=1) * weight).mean()

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

        self.current_coupling = self._build_reciprocal_coupling(direction)
        progress.update(1)
        progress.set_postfix(phase=f"{outer_iteration}:{direction}", loss=f"{(running_loss or 0.0):.5f}")

    def train(
        self,
        *,
        checkpoint_schedule: UpdateQualityCheckpointSchedule | None = None,
        checkpoint_callback: Callable[[int, "PairwiseDSBMTrainer"], None] | None = None,
    ) -> None:
        total_steps = 2 * self.hyperparams.n_outer
        self.completed_optimizer_updates = 0
        progress = tqdm(total=total_steps, desc=self.progress_label, leave=False, dynamic_ncols=True)
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

    @torch.inference_mode()
    def sample_probability_flow(
        self,
        source_points: torch.Tensor,
        *,
        rollout_batch_size: int,
    ) -> torch.Tensor:
        steps = self._num_rollout_steps()
        dt = 1.0 / steps
        outputs = []
        forward_net = self.net["f"]
        backward_net = self.net["b"]
        fw_training = forward_net.training
        bw_training = backward_net.training
        forward_net.eval()
        backward_net.eval()
        try:
            for start in range(0, source_points.shape[0], rollout_batch_size):
                x = source_points[start: start + rollout_batch_size].to(self.device)
                for step in range(steps):
                    current_t = step * dt
                    t = torch.full((x.shape[0],), current_t, dtype=x.dtype, device=x.device)
                    x = x + dt * 0.5 * (forward_net(x, t) - backward_net(x, t))
                outputs.append(x.cpu())
        finally:
            if fw_training:
                forward_net.train()
            if bw_training:
                backward_net.train()
        return torch.cat(outputs, dim=0)


def resolve_effective_num_iter(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    *,
    hyperparams: DSBMHyperparams,
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
        return manual_num_iter, sf2m_steps_per_epoch, 2 * hyperparams.n_outer * manual_num_iter
    num_iter = max(1, math.ceil(sf2m_total_updates / (2 * hyperparams.n_outer)))
    return num_iter, sf2m_steps_per_epoch, 2 * hyperparams.n_outer * num_iter


def train_pairwise_dsbm(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    *,
    hyperparams: DSBMHyperparams,
    sf2m_epochs: int,
    manual_num_iter: int | None,
    device: torch.device,
    progress_label: str,
    quality_eval_every: int,
    checkpoint_callback: Callable[[int, PairwiseDSBMTrainer], None] | None = None,
) -> PairwiseDSBMTrainer:
    effective_num_iter, steps_per_epoch, total_updates = resolve_effective_num_iter(
        source_points,
        target_points,
        hyperparams=hyperparams,
        sf2m_epochs=sf2m_epochs,
        manual_num_iter=manual_num_iter,
    )
    trainer = PairwiseDSBMTrainer(
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
    trainer: PairwiseDSBMTrainer,
    source_points: torch.Tensor,
    *,
    rollout_batch_size: int,
) -> torch.Tensor:
    return trainer.sample_probability_flow(source_points, rollout_batch_size=rollout_batch_size)
