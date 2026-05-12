from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
from torchcfm.models import MLP
from torchcfm.optimal_transport import OTPlanSampler
from tqdm import tqdm

from toy_exp.common import compute_reference_updates, sample_minibatch
from toy_exp.sf2m import make_time_input


class TrajectoryNetVectorField(nn.Module):
    def __init__(self, *, dim: int, width: int, device: torch.device) -> None:
        super().__init__()
        self.net = MLP(dim=dim, time_varying=True, w=width).to(device)

    def velocity(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(make_time_input(x, t))


def rk4_integrate_interval(
    vector_field: TrajectoryNetVectorField,
    state: torch.Tensor,
    *,
    steps: int,
    track_energy: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    dt = 1.0 / steps
    x = state
    total_energy = x.new_zeros(())
    for step in range(steps):
        current_t = step * dt
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
    return x, total_energy if track_energy else None


@torch.inference_mode()
def sample_predicted_target(
    vector_field: TrajectoryNetVectorField,
    source_points: torch.Tensor,
    *,
    steps_per_unit: int,
    rollout_batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    steps = max(1, int(math.ceil(steps_per_unit)))
    outputs = []
    vector_field.eval()
    for start in range(0, source_points.shape[0], rollout_batch_size):
        batch = source_points[start: start + rollout_batch_size].to(device)
        out, _ = rk4_integrate_interval(
            vector_field,
            batch,
            steps=steps,
            track_energy=False,
        )
        outputs.append(out.cpu())
    return torch.cat(outputs, dim=0)


def train_pairwise_trajectory_net(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    *,
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
    dim = int(source_points.shape[1])
    vector_field = TrajectoryNetVectorField(dim=dim, width=width, device=device)
    optimizer = torch.optim.AdamW(vector_field.parameters(), lr=lr, weight_decay=weight_decay)
    ot_sampler = OTPlanSampler(method=ot_method)
    _, steps_per_epoch = compute_reference_updates(
        source_points,
        target_points,
        batch_size=batch_size,
        epochs=epochs,
    )
    ode_steps = max(1, int(math.ceil(steps_per_unit)))
    progress = tqdm(range(epochs), desc=progress_label, leave=False, dynamic_ncols=True)
    running_loss = None

    for epoch_idx in progress:
        epoch_loss = 0.0
        n_updates = 0
        vector_field.train()
        for _ in range(steps_per_epoch):
            x0 = sample_minibatch(source_points, batch_size)
            x1 = sample_minibatch(target_points, batch_size)
            x0, x1 = ot_sampler.sample_plan(x0, x1)
            x0 = x0.to(device)
            x1 = x1.to(device)
            predicted, energy = rk4_integrate_interval(
                vector_field,
                x0,
                steps=ode_steps,
                track_energy=True,
            )
            endpoint_loss = torch.mean((predicted - x1) ** 2)
            loss = endpoint_loss + energy_weight * energy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            n_updates += 1

        mean_loss = epoch_loss / max(n_updates, 1)
        running_loss = mean_loss if running_loss is None else 0.95 * running_loss + 0.05 * mean_loss
        progress.set_postfix(loss=f"{running_loss:.5f}")
        epoch_number = epoch_idx + 1
        if checkpoint_callback is not None and quality_eval_every > 0 and (
            epoch_number % quality_eval_every == 0 or epoch_number == epochs
        ):
            checkpoint_callback(epoch_number, vector_field)

    return vector_field
