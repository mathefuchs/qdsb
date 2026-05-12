from __future__ import annotations

import math

import numpy as np
import torch
from cell_exp.common import find_bracketing_interval, subsample_points
from cell_exp.quality import compute_mmd
from torchcfm.models import MLP
from torchcfm.optimal_transport import wasserstein


def make_time_input(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, t[:, None]], dim=1)


def build_models(dim: int, width: int, device: torch.device) -> tuple[MLP, MLP]:
    flow_model = MLP(dim=dim, time_varying=True, w=width).to(device)
    score_model = MLP(dim=dim, time_varying=True, w=width).to(device)
    return flow_model, score_model


def rollout_sde_interval(
    flow_model: MLP,
    score_model: MLP,
    state: torch.Tensor,
    *,
    start_time: float,
    end_time: float,
    sigma: float,
    steps: int,
) -> torch.Tensor:
    dt = (end_time - start_time) / steps
    sqrt_dt = math.sqrt(dt)
    x = state
    for step in range(steps):
        current_t = start_time + step * dt
        t = torch.full((x.shape[0],), current_t,
                       dtype=x.dtype, device=x.device)
        x_with_t = make_time_input(x, t)
        drift = flow_model(x_with_t) + score_model(x_with_t)
        x = x + dt * drift + sigma * sqrt_dt * torch.randn_like(x)
    return x


def propagate_to_missing_time(
    flow_model: MLP,
    score_model: MLP,
    source_points: torch.Tensor,
    *,
    start_time: float,
    target_time: float,
    sigma: float,
    steps_per_unit: int,
    rollout_batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    duration = target_time - start_time
    if duration <= 0:
        raise ValueError("target_time must be larger than start_time.")

    steps = max(1, int(math.ceil(duration * steps_per_unit)))
    outputs = []
    flow_model.eval()
    score_model.eval()
    with torch.inference_mode():
        for start in range(0, source_points.shape[0], rollout_batch_size):
            batch = source_points[start: start + rollout_batch_size].to(device)
            out = rollout_sde_interval(
                flow_model,
                score_model,
                batch,
                start_time=start_time,
                end_time=target_time,
                sigma=sigma,
                steps=steps,
            )
            outputs.append(out.cpu())
    return torch.cat(outputs, dim=0)


def evaluate_leave_one_out(
    flow_model: MLP,
    score_model: MLP,
    timepoints: list[torch.Tensor],
    model_times: np.ndarray,
    *,
    missing_index: int,
    sigma: float,
    steps_per_unit: int,
    rollout_batch_size: int,
    device: torch.device,
    max_eval_points: int | None,
    w1_method: str,
    w1_reg: float,
) -> float:
    observed_indices = [idx for idx in range(
        len(timepoints)) if idx != missing_index]
    left_idx, _ = find_bracketing_interval(observed_indices, missing_index)
    predicted = propagate_to_missing_time(
        flow_model,
        score_model,
        timepoints[left_idx],
        start_time=float(model_times[left_idx]),
        target_time=float(model_times[missing_index]),
        sigma=sigma,
        steps_per_unit=steps_per_unit,
        rollout_batch_size=rollout_batch_size,
        device=device,
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


def evaluate_leave_one_out_mmd(
    flow_model: MLP,
    score_model: MLP,
    timepoints: list[torch.Tensor],
    model_times: np.ndarray,
    *,
    missing_index: int,
    sigma: float,
    steps_per_unit: int,
    rollout_batch_size: int,
    device: torch.device,
    gamma: float,
    max_eval_points: int | None,
) -> float:
    observed_indices = [idx for idx in range(
        len(timepoints)) if idx != missing_index]
    left_idx, _ = find_bracketing_interval(observed_indices, missing_index)
    predicted = propagate_to_missing_time(
        flow_model,
        score_model,
        timepoints[left_idx],
        start_time=float(model_times[left_idx]),
        target_time=float(model_times[missing_index]),
        sigma=sigma,
        steps_per_unit=steps_per_unit,
        rollout_batch_size=rollout_batch_size,
        device=device,
    )
    return compute_mmd(
        predicted,
        timepoints[missing_index],
        gamma=gamma,
        max_points=max_eval_points,
    )
