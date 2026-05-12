from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from img_exp.common import compute_reference_updates, sample_minibatch
from img_exp.quality import UpdateQualityCheckpointSchedule
from torch.utils.data import DataLoader, Dataset
from torchcfm.conditional_flow_matching import SchrodingerBridgeConditionalFlowMatcher
from torchcfm.models import MLP
from torchcfm.optimal_transport import OTPlanSampler
from tqdm import tqdm

DEFAULT_MIN_COV_SCALE = 1e-2
DEFAULT_TIME_EPS = 1e-4
DEFAULT_LIGHTSB_M_POINT_CHUNK_SIZE = 64
DEFAULT_LIGHTSB_M_CHOLESKY_JITTER = 1e-6
DEFAULT_LIGHTSB_M_MAX_CHOLESKY_TRIES = 8


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
    sigma: float,
    steps: int,
) -> torch.Tensor:
    dt = 1.0 / steps
    sqrt_dt = math.sqrt(dt)
    x = state
    for step in range(steps):
        current_t = step * dt
        t = torch.full((x.shape[0],), current_t, dtype=x.dtype, device=x.device)
        xt = make_time_input(x, t)
        drift = flow_model(xt) + score_model(xt)
        x = x + dt * drift + sigma * sqrt_dt * torch.randn_like(x)
    return x


@torch.inference_mode()
def sample_predicted_target_qdsb(
    flow_model: MLP,
    score_model: MLP,
    source_points: torch.Tensor,
    *,
    sigma: float,
    steps_per_unit: int,
    rollout_batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    steps = max(1, int(math.ceil(steps_per_unit)))
    outputs = []
    flow_model.eval()
    score_model.eval()
    for start in range(0, source_points.shape[0], rollout_batch_size):
        batch = source_points[start: start + rollout_batch_size].to(device)
        outputs.append(
            rollout_sde_interval(
                flow_model,
                score_model,
                batch,
                sigma=sigma,
                steps=steps,
            ).cpu()
        )
    return torch.cat(outputs, dim=0)


def train_pairwise_sf2m(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    *,
    sigma: float,
    ot_method: str,
    batch_size: int,
    epochs: int,
    train_seconds: float | None,
    width: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    progress_label: str,
    quality_eval_every: int,
    checkpoint_callback: Callable[[int, MLP, MLP], None] | None = None,
    epoch_callback: Callable[[int, MLP, MLP], bool | None] | None = None,
) -> tuple[MLP, MLP, int]:
    dim = int(source_points.shape[1])
    flow_model, score_model = build_models(dim=dim, width=width, device=device)
    optimizer = torch.optim.AdamW(
        list(flow_model.parameters()) + list(score_model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    flow_matcher = SchrodingerBridgeConditionalFlowMatcher(sigma=sigma, ot_method=ot_method)
    _, steps_per_epoch = compute_reference_updates(
        source_points,
        target_points,
        batch_size=batch_size,
        epochs=1,
    )
    total_epochs = epochs if epochs > 0 else None
    progress = tqdm(total=total_epochs, desc=progress_label, leave=False, dynamic_ncols=True)
    running_loss = None
    epoch_idx = 0

    while total_epochs is None or epoch_idx < total_epochs:
        epoch_loss = 0.0
        n_updates = 0
        flow_model.train()
        score_model.train()
        for _ in range(steps_per_epoch):
            x0 = sample_minibatch(source_points, batch_size).to(device)
            x1 = sample_minibatch(target_points, batch_size).to(device)
            tau, xt, ut, eps = flow_matcher.sample_location_and_conditional_flow(
                x0,
                x1,
                return_noise=True,
            )
            xt_with_t = make_time_input(xt, tau)
            vt = flow_model(xt_with_t)
            st = score_model(xt_with_t)
            lambda_t = flow_matcher.compute_lambda(tau)
            loss = torch.mean((vt - ut) ** 2) + torch.mean((lambda_t[:, None] * st + eps) ** 2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            n_updates += 1

        mean_loss = epoch_loss / max(n_updates, 1)
        running_loss = mean_loss if running_loss is None else 0.95 * running_loss + 0.05 * mean_loss
        progress.update(1)
        progress.set_postfix(loss=f"{running_loss:.5f}")

        epoch_number = epoch_idx + 1
        if checkpoint_callback is not None and quality_eval_every > 0 and (
            epoch_number % quality_eval_every == 0 or epoch_number == total_epochs
        ):
            checkpoint_callback(epoch_number, flow_model, score_model)
        should_stop = False
        if epoch_callback is not None:
            should_stop = bool(epoch_callback(epoch_number, flow_model, score_model))
        epoch_idx += 1
        if should_stop:
            break

    progress.close()
    return flow_model, score_model, epoch_idx


@dataclass(frozen=True)
class CoverageAccelerationPlan:
    anchor_plan: torch.Tensor
    source_anchors: torch.Tensor
    target_anchors: torch.Tensor
    source_assignments: torch.Tensor
    target_assignments: torch.Tensor
    source_anchor_to_point_ind: dict[int, list[int]]
    target_anchor_to_point_ind: dict[int, list[int]]
    matched_point_plans: dict[tuple[int, int], "PointMatchPlan"] | None = None


@dataclass(frozen=True)
class PointMatchPlan:
    pair_probs: torch.Tensor
    source_point_indices: torch.Tensor
    target_point_indices: torch.Tensor


def farthest_first_k_center(
    dataset: torch.Tensor,
    k: int,
    *,
    initial_center: int | None = None,
) -> torch.Tensor:
    effective_k = min(k, dataset.shape[0])
    if initial_center is None:
        initial_center = random.randrange(dataset.shape[0])
    centers = [initial_center]
    min_distances = torch.cdist(dataset[[initial_center]], dataset).squeeze(0)
    for _ in range(1, effective_k):
        next_center = torch.argmax(min_distances).item()
        centers.append(next_center)
        distances_to_new_center = torch.cdist(dataset[[next_center]], dataset).squeeze(0)
        min_distances = torch.minimum(min_distances, distances_to_new_center)
    return dataset[centers].clone()


def anchor_cover_radius(dataset: torch.Tensor, anchors: torch.Tensor) -> float:
    distances = torch.cdist(dataset, anchors)
    return float(distances.min(dim=1).values.max().item())


def gon_plus_k_center(
    dataset: torch.Tensor,
    k: int,
    *,
    num_candidates: int,
) -> torch.Tensor:
    effective_candidates = min(num_candidates, dataset.shape[0])
    candidate_indices = random.sample(range(dataset.shape[0]), k=effective_candidates)
    best_anchors = None
    best_radius = float("inf")
    for candidate_index in candidate_indices:
        anchors = farthest_first_k_center(dataset, k=k, initial_center=candidate_index)
        radius = anchor_cover_radius(dataset, anchors)
        if radius < best_radius:
            best_radius = radius
            best_anchors = anchors
    if best_anchors is None:
        raise RuntimeError("Gon+ failed to produce anchors.")
    return best_anchors


def anchors_and_weights(
    dataset: torch.Tensor,
    k: int,
    *,
    method: str,
    gon_plus_candidates: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if method == "gon":
        anchors = farthest_first_k_center(dataset, k=k)
    elif method == "gon_plus":
        anchors = gon_plus_k_center(dataset, k=k, num_candidates=gon_plus_candidates)
    else:
        raise ValueError(f"Unsupported anchor selection method: {method}")
    distances = torch.cdist(dataset, anchors)
    assignments = torch.argmin(distances, dim=1)
    counts = torch.bincount(assignments, minlength=anchors.shape[0])
    weights = counts.float() / dataset.shape[0]
    return anchors, weights, assignments


def temperature_scale_probs(probs: torch.Tensor, temperature: float) -> torch.Tensor:
    probs = probs.to(dtype=torch.float32).clamp_min(1e-12)
    if temperature == 1.0:
        return probs / probs.sum().clamp_min(1e-12)
    scaled = probs.pow(1.0 / temperature)
    return scaled / scaled.sum().clamp_min(1e-12)


def sample_anchors(
    anchor_probs: torch.Tensor,
    batch_size: int,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    p = anchor_probs.flatten().clamp_min(0)
    p = temperature_scale_probs(p / p.sum().clamp_min(1e-12), temperature)
    choices = torch.multinomial(p, num_samples=batch_size, replacement=True)
    return (
        torch.div(choices, anchor_probs.shape[1], rounding_mode="floor"),
        torch.remainder(choices, anchor_probs.shape[1]),
    )


def anchor_cell_map(assignments: torch.Tensor) -> dict[int, list[int]]:
    anchor_to_point_ind: dict[int, list[int]] = {}
    for point_index, anchor_index in enumerate(assignments.tolist()):
        anchor_to_point_ind.setdefault(anchor_index, []).append(point_index)
    return anchor_to_point_ind


def anchor_idx_to_rand_point_in_cell(
    anchor_ind: torch.Tensor,
    anchor_to_point_ind: dict[int, list[int]],
    dataset: torch.Tensor,
) -> torch.Tensor:
    point_indices = [random.choice(anchor_to_point_ind[int(anchor_idx)]) for anchor_idx in anchor_ind.tolist()]
    return dataset[torch.tensor(point_indices, dtype=torch.long, device=dataset.device)]


def solve_ot_plan(
    source_weights: torch.Tensor,
    target_weights: torch.Tensor,
    cost: torch.Tensor,
    *,
    flow_matcher: SchrodingerBridgeConditionalFlowMatcher,
) -> torch.Tensor:
    if flow_matcher.ot_sampler.normalize_cost:
        cost = cost / cost.max().clamp_min(1e-12)
    plan = flow_matcher.ot_sampler.ot_fn(
        source_weights.cpu().numpy(),
        target_weights.cpu().numpy(),
        cost.cpu().numpy(),
    )
    plan = torch.from_numpy(plan).to(dtype=torch.float32)
    plan = plan.clamp_min_(0)
    if float(plan.sum()) <= 1e-12:
        raise RuntimeError("Coverage OT plan is degenerate.")
    return plan


def build_point_match_plan(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    *,
    source_point_indices: list[int],
    target_point_indices: list[int],
    flow_matcher: SchrodingerBridgeConditionalFlowMatcher,
) -> PointMatchPlan:
    source_subset = source_points[source_point_indices]
    target_subset = target_points[target_point_indices]
    source_weights = torch.full((source_subset.shape[0],), 1.0 / source_subset.shape[0], dtype=torch.float32)
    target_weights = torch.full((target_subset.shape[0],), 1.0 / target_subset.shape[0], dtype=torch.float32)
    point_plan = solve_ot_plan(
        source_weights,
        target_weights,
        torch.cdist(source_subset, target_subset) ** 2,
        flow_matcher=flow_matcher,
    )
    support = torch.nonzero(point_plan > 0, as_tuple=False)
    if support.numel() == 0:
        raise RuntimeError("Point-matching produced an empty support.")
    pair_probs = point_plan[support[:, 0], support[:, 1]]
    pair_probs = pair_probs / pair_probs.sum().clamp_min(1e-12)
    source_indices = torch.tensor(source_point_indices, dtype=torch.long)[support[:, 0]]
    target_indices = torch.tensor(target_point_indices, dtype=torch.long)[support[:, 1]]
    return PointMatchPlan(
        pair_probs=pair_probs,
        source_point_indices=source_indices,
        target_point_indices=target_indices,
    )


def sample_matched_points_from_ot_plans(
    source_anchor_ind: torch.Tensor,
    target_anchor_ind: torch.Tensor,
    *,
    coverage_plan: CoverageAccelerationPlan,
    source_dataset: torch.Tensor,
    target_dataset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if coverage_plan.matched_point_plans is None:
        raise RuntimeError("Matched point plans were requested but not built.")
    num_target_anchors = coverage_plan.anchor_plan.shape[1]
    pair_codes = source_anchor_ind * num_target_anchors + target_anchor_ind
    sample_count = source_anchor_ind.shape[0]
    source_indices = torch.empty(sample_count, dtype=torch.long)
    target_indices = torch.empty(sample_count, dtype=torch.long)
    unique_codes, inverse = torch.unique(pair_codes, return_inverse=True)
    for unique_offset, code in enumerate(unique_codes.tolist()):
        mask = inverse == unique_offset
        num_samples = int(mask.sum().item())
        source_anchor = code // num_target_anchors
        target_anchor = code % num_target_anchors
        point_plan = coverage_plan.matched_point_plans[(source_anchor, target_anchor)]
        choices = torch.multinomial(point_plan.pair_probs, num_samples=num_samples, replacement=True)
        source_indices[mask] = point_plan.source_point_indices[choices]
        target_indices[mask] = point_plan.target_point_indices[choices]
    return source_dataset[source_indices], target_dataset[target_indices]


def build_coverage_acceleration_plan(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    *,
    num_anchors: int,
    anchor_selection: str,
    gon_plus_candidates: int,
    flow_matcher: SchrodingerBridgeConditionalFlowMatcher,
    point_match_mode: str,
) -> CoverageAccelerationPlan:
    source_anchors, source_weights, source_assignments = anchors_and_weights(
        source_points,
        k=num_anchors,
        method=anchor_selection,
        gon_plus_candidates=gon_plus_candidates,
    )
    target_anchors, target_weights, target_assignments = anchors_and_weights(
        target_points,
        k=num_anchors,
        method=anchor_selection,
        gon_plus_candidates=gon_plus_candidates,
    )
    anchor_plan = solve_ot_plan(
        source_weights,
        target_weights,
        torch.cdist(source_anchors, target_anchors) ** 2,
        flow_matcher=flow_matcher,
    )
    source_anchor_to_point_ind = anchor_cell_map(source_assignments)
    target_anchor_to_point_ind = anchor_cell_map(target_assignments)
    matched_point_plans = None
    if point_match_mode == "ot":
        matched_point_plans = {}
        for source_anchor, target_anchor in torch.nonzero(anchor_plan > 0, as_tuple=False).tolist():
            matched_point_plans[(source_anchor, target_anchor)] = build_point_match_plan(
                source_points,
                target_points,
                source_point_indices=source_anchor_to_point_ind[source_anchor],
                target_point_indices=target_anchor_to_point_ind[target_anchor],
                flow_matcher=flow_matcher,
            )
    return CoverageAccelerationPlan(
        anchor_plan=anchor_plan,
        source_anchors=source_anchors,
        target_anchors=target_anchors,
        source_assignments=source_assignments,
        target_assignments=target_assignments,
        source_anchor_to_point_ind=source_anchor_to_point_ind,
        target_anchor_to_point_ind=target_anchor_to_point_ind,
        matched_point_plans=matched_point_plans,
    )


def train_pairwise_qdsb(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    *,
    coverage_anchors: int,
    coverage_anchor_selection: str,
    coverage_anchor_gon_plus_candidates: int,
    coverage_anchor_weight_temperature: float,
    coverage_anchor_refresh_epochs: int,
    coverage_point_match_mode: str,
    sigma: float,
    ot_method: str,
    batch_size: int,
    epochs: int,
    train_seconds: float | None,
    width: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    progress_label: str,
    quality_eval_every: int,
    checkpoint_callback: Callable[[int, MLP, MLP], None] | None = None,
    epoch_callback: Callable[[int, MLP, MLP], bool | None] | None = None,
) -> tuple[MLP, MLP, int]:
    dim = int(source_points.shape[1])
    flow_model, score_model = build_models(dim=dim, width=width, device=device)
    optimizer = torch.optim.AdamW(
        list(flow_model.parameters()) + list(score_model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    flow_matcher = SchrodingerBridgeConditionalFlowMatcher(sigma=sigma, ot_method=ot_method)

    def refresh_coverage_plan() -> CoverageAccelerationPlan:
        return build_coverage_acceleration_plan(
            source_points,
            target_points,
            num_anchors=coverage_anchors,
            anchor_selection=coverage_anchor_selection,
            gon_plus_candidates=coverage_anchor_gon_plus_candidates,
            flow_matcher=flow_matcher,
            point_match_mode=coverage_point_match_mode,
        )

    coverage_plan = refresh_coverage_plan()
    _, steps_per_epoch = compute_reference_updates(
        source_points,
        target_points,
        batch_size=batch_size,
        epochs=1,
    )
    total_epochs = epochs if epochs > 0 else None
    progress = tqdm(total=total_epochs, desc=progress_label, leave=False, dynamic_ncols=True)
    running_loss = None
    epoch_idx = 0
    while total_epochs is None or epoch_idx < total_epochs:
        if (
            coverage_anchor_refresh_epochs > 0
            and epoch_idx > 0
            and epoch_idx % coverage_anchor_refresh_epochs == 0
        ):
            coverage_plan = refresh_coverage_plan()

        epoch_loss = 0.0
        n_updates = 0
        flow_model.train()
        score_model.train()
        for _ in range(steps_per_epoch):
            sampled_source_anchors, sampled_target_anchors = sample_anchors(
                coverage_plan.anchor_plan,
                batch_size=batch_size,
                temperature=coverage_anchor_weight_temperature,
            )
            if coverage_point_match_mode == "ot":
                x0, x1 = sample_matched_points_from_ot_plans(
                    sampled_source_anchors,
                    sampled_target_anchors,
                    coverage_plan=coverage_plan,
                    source_dataset=source_points,
                    target_dataset=target_points,
                )
            elif coverage_point_match_mode == "random":
                x0 = anchor_idx_to_rand_point_in_cell(
                    sampled_source_anchors,
                    coverage_plan.source_anchor_to_point_ind,
                    source_points,
                )
                x1 = anchor_idx_to_rand_point_in_cell(
                    sampled_target_anchors,
                    coverage_plan.target_anchor_to_point_ind,
                    target_points,
                )
            else:
                raise ValueError(f"Unsupported coverage point match mode: {coverage_point_match_mode}")

            x0 = x0.to(device)
            x1 = x1.to(device)
            tau, xt, ut, eps = flow_matcher.sample_location_and_conditional_flow(
                x0,
                x1,
                return_noise=True,
                use_ot=False,
            )
            xt_with_t = make_time_input(xt, tau)
            vt = flow_model(xt_with_t)
            st = score_model(xt_with_t)
            lambda_t = flow_matcher.compute_lambda(tau)
            loss = torch.mean((vt - ut) ** 2) + torch.mean((lambda_t[:, None] * st + eps) ** 2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            n_updates += 1

        mean_loss = epoch_loss / max(n_updates, 1)
        running_loss = mean_loss if running_loss is None else 0.95 * running_loss + 0.05 * mean_loss
        progress.update(1)
        progress.set_postfix(loss=f"{running_loss:.5f}")

        epoch_number = epoch_idx + 1
        if checkpoint_callback is not None and quality_eval_every > 0 and (
            epoch_number % quality_eval_every == 0 or epoch_number == total_epochs
        ):
            checkpoint_callback(epoch_number, flow_model, score_model)
        should_stop = False
        if epoch_callback is not None:
            should_stop = bool(epoch_callback(epoch_number, flow_model, score_model))
        epoch_idx += 1
        if should_stop:
            break

    progress.close()
    return flow_model, score_model, epoch_idx


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
        # Keep cached bridge samples on CPU; training moves minibatches to the active device.
        self.data = torch.zeros(
            (num_batches, batch_size * langevin.num_steps, 2, langevin.x_dim),
            device="cpu",
        )
        self.steps_data = torch.zeros(
            (num_batches, batch_size * langevin.num_steps, 1),
            device="cpu",
        )

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
        self.gammas = torch.tensor(gammas, device=device, dtype=torch.float32)
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
        self.completed_optimizer_updates = 0

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
        stop_callback: Callable[[int], bool] | None = None,
    ) -> bool:
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
            if stop_callback is not None and stop_callback(self.completed_optimizer_updates):
                progress.set_postfix(phase=f"{ipf_iteration}:{forward_or_backward}", loss=f"{(running_loss or 0.0):.5f}")
                return True

        progress.update(1)
        progress.set_postfix(phase=f"{ipf_iteration}:{forward_or_backward}", loss=f"{(running_loss or 0.0):.5f}")
        return False

    def train(
        self,
        *,
        checkpoint_schedule: UpdateQualityCheckpointSchedule | None = None,
        checkpoint_callback: Callable[[int, "PairwiseDSBTrainer"], None] | None = None,
        stop_callback: Callable[[int], bool] | None = None,
    ) -> None:
        progress = tqdm(desc=self.progress_label, leave=False, dynamic_ncols=True)
        ipf_iteration = 1
        try:
            while True:
                stop = self.ipf_step(
                    "b",
                    ipf_iteration,
                    progress=progress,
                    checkpoint_schedule=checkpoint_schedule,
                    checkpoint_callback=checkpoint_callback,
                    stop_callback=stop_callback,
                )
                if stop:
                    break
                stop = self.ipf_step(
                    "f",
                    ipf_iteration,
                    progress=progress,
                    checkpoint_schedule=checkpoint_schedule,
                    checkpoint_callback=checkpoint_callback,
                    stop_callback=stop_callback,
                )
                if stop:
                    break
                ipf_iteration += 1
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


def train_pairwise_dsb(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    *,
    hyperparams: DSBHyperparams,
    epochs: int,
    train_seconds: float | None,
    device: torch.device,
    progress_label: str,
    quality_eval_every: int,
    checkpoint_callback: Callable[[int, PairwiseDSBTrainer], None] | None = None,
    stop_callback: Callable[[int], bool] | None = None,
) -> tuple[PairwiseDSBTrainer, int]:
    steps_per_epoch = compute_reference_updates(
        source_points,
        target_points,
        batch_size=hyperparams.batch_size,
        epochs=1,
    )[1]
    total_updates = None if epochs <= 0 else epochs * steps_per_epoch
    trainer = PairwiseDSBTrainer(
        source_points=source_points,
        target_points=target_points,
        device=device,
        hyperparams=hyperparams,
        progress_label=progress_label,
    )
    schedule = None
    if checkpoint_callback is not None and quality_eval_every > 0:
        schedule = UpdateQualityCheckpointSchedule(
            steps_per_epoch=steps_per_epoch,
            eval_every_epochs=quality_eval_every,
            total_updates=total_updates,
        )
    trainer.train(
        checkpoint_schedule=schedule,
        checkpoint_callback=checkpoint_callback,
        stop_callback=stop_callback,
    )
    completed_epochs = max(1, int(math.ceil(trainer.completed_optimizer_updates / steps_per_epoch)))
    return trainer, completed_epochs


@torch.inference_mode()
def sample_predicted_target_dsb(
    trainer: PairwiseDSBTrainer,
    source_points: torch.Tensor,
    *,
    rollout_batch_size: int,
) -> torch.Tensor:
    return trainer.sample_forward_population(source_points, rollout_batch_size=rollout_batch_size)


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
        self.completed_optimizer_updates = 0

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
        stop_callback: Callable[[int], bool] | None = None,
    ) -> bool:
        net = self.net[direction]
        optimizer = self.optimizer[direction]
        net.train()
        running_loss = None
        for _ in range(self.hyperparams.num_iter):
            endpoint_source, endpoint_target = self._sample_training_endpoints()
            endpoint_source = endpoint_source.to(self.device)
            endpoint_target = endpoint_target.to(self.device)
            tau, xt, forward_target, backward_target, forward_weight, backward_weight = sample_brownian_bridge_batch(
                endpoint_source,
                endpoint_target,
                sigma=self.hyperparams.sigma,
                loss_weighting=self.hyperparams.loss_weighting,
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
            if stop_callback is not None and stop_callback(self.completed_optimizer_updates):
                progress.set_postfix(phase=f"{outer_iteration}:{direction}", loss=f"{(running_loss or 0.0):.5f}")
                return True

        self.current_coupling = self._build_reciprocal_coupling(direction)
        progress.update(1)
        progress.set_postfix(phase=f"{outer_iteration}:{direction}", loss=f"{(running_loss or 0.0):.5f}")
        return False

    def train(
        self,
        *,
        checkpoint_schedule: UpdateQualityCheckpointSchedule | None = None,
        checkpoint_callback: Callable[[int, "PairwiseDSBMTrainer"], None] | None = None,
        stop_callback: Callable[[int], bool] | None = None,
    ) -> None:
        progress = tqdm(desc=self.progress_label, leave=False, dynamic_ncols=True)
        outer_iteration = 1
        try:
            while True:
                stop = self._train_phase(
                    "b",
                    outer_iteration,
                    progress=progress,
                    checkpoint_schedule=checkpoint_schedule,
                    checkpoint_callback=checkpoint_callback,
                    stop_callback=stop_callback,
                )
                if stop:
                    break
                stop = self._train_phase(
                    "f",
                    outer_iteration,
                    progress=progress,
                    checkpoint_schedule=checkpoint_schedule,
                    checkpoint_callback=checkpoint_callback,
                    stop_callback=stop_callback,
                )
                if stop:
                    break
                outer_iteration += 1
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


def train_pairwise_dsbm(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    *,
    hyperparams: DSBMHyperparams,
    epochs: int,
    train_seconds: float | None,
    device: torch.device,
    progress_label: str,
    quality_eval_every: int,
    checkpoint_callback: Callable[[int, PairwiseDSBMTrainer], None] | None = None,
    stop_callback: Callable[[int], bool] | None = None,
) -> tuple[PairwiseDSBMTrainer, int]:
    steps_per_epoch = compute_reference_updates(
        source_points,
        target_points,
        batch_size=hyperparams.batch_size,
        epochs=1,
    )[1]
    total_updates = None if epochs <= 0 else epochs * steps_per_epoch
    trainer = PairwiseDSBMTrainer(
        source_points=source_points,
        target_points=target_points,
        device=device,
        hyperparams=hyperparams,
        progress_label=progress_label,
    )
    schedule = None
    if checkpoint_callback is not None and quality_eval_every > 0:
        schedule = UpdateQualityCheckpointSchedule(
            steps_per_epoch=steps_per_epoch,
            eval_every_epochs=quality_eval_every,
            total_updates=total_updates,
        )
    trainer.train(
        checkpoint_schedule=schedule,
        checkpoint_callback=checkpoint_callback,
        stop_callback=stop_callback,
    )
    completed_epochs = max(1, int(math.ceil(trainer.completed_optimizer_updates / steps_per_epoch)))
    return trainer, completed_epochs


@torch.inference_mode()
def sample_predicted_target_dsbm(
    trainer: PairwiseDSBMTrainer,
    source_points: torch.Tensor,
    *,
    rollout_batch_size: int,
) -> torch.Tensor:
    return trainer.sample_probability_flow(source_points, rollout_batch_size=rollout_batch_size)


def inverse_softplus(value: float) -> float:
    if value <= 0:
        raise ValueError("softplus inverse expects a positive value.")
    return float(math.log(math.expm1(value)))


def stable_cholesky(
    matrix: torch.Tensor,
    *,
    jitter_scale: float = DEFAULT_LIGHTSB_M_CHOLESKY_JITTER,
    max_tries: int = DEFAULT_LIGHTSB_M_MAX_CHOLESKY_TRIES,
) -> torch.Tensor:
    # The theoretical system is SPD, but finite-precision batches in 512-d latents can drift
    # slightly off-symmetry / off-PD. Symmetrize first, then add adaptive diagonal jitter.
    symmetric = 0.5 * (matrix + matrix.transpose(-1, -2))
    eye = torch.eye(
        symmetric.shape[-1],
        device=symmetric.device,
        dtype=symmetric.dtype,
    )
    diagonal_scale = torch.diagonal(symmetric, dim1=-2, dim2=-1).abs().mean().clamp_min(1.0)
    jitter = symmetric.new_tensor(jitter_scale) * diagonal_scale
    latest_info = None
    for _ in range(max_tries):
        candidate = symmetric + jitter * eye
        chol, info = torch.linalg.cholesky_ex(candidate, check_errors=False)
        if bool((info == 0).all().item()):
            return chol
        latest_info = info
        jitter = jitter * 10.0
    failing = int(latest_info[latest_info > 0][0].item()) if latest_info is not None and bool((latest_info > 0).any().item()) else -1
    raise RuntimeError(
        "LightSB-M Cholesky factorization failed after adaptive jittering. "
        f"Last failing leading minor order: {failing}."
    )


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
        indices = torch.randint(
            0,
            init_points.shape[0],
            (num_components,),
            device=init_points.device,
        )
        means_init = init_points[indices].clone()
        var = init_points.var(dim=0, unbiased=False)
        avg_std = float(torch.sqrt(var.mean().clamp_min(1e-6)).item())
        diag_init = max(avg_std, min_cov_scale)
        raw_diag_init = inverse_softplus(diag_init - min_cov_scale)
        raw_tril = torch.zeros(
            num_components,
            dim,
            dim,
            dtype=init_points.dtype,
            device=init_points.device,
        )
        diag_indices = torch.arange(dim, device=init_points.device)
        raw_tril[:, diag_indices, diag_indices] = raw_diag_init

        self.dim = dim
        self.min_cov_scale = min_cov_scale
        self.logits = nn.Parameter(
            torch.zeros(
                num_components,
                dtype=init_points.dtype,
                device=init_points.device,
            )
        )
        self.means = nn.Parameter(means_init)
        self.raw_tril = nn.Parameter(raw_tril)

    def scale_tril(self) -> torch.Tensor:
        lower = torch.tril(self.raw_tril, diagonal=-1)
        diag = F.softplus(torch.diagonal(self.raw_tril, dim1=-2, dim2=-1)) + self.min_cov_scale
        return lower + torch.diag_embed(diag)

    def component_stats(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scale_tril = self.scale_tril()
        covariance = scale_tril @ scale_tril.transpose(-1, -2)
        precision = torch.cholesky_inverse(scale_tril)
        precision = 0.5 * (precision + precision.transpose(-1, -2))
        logdet_cov = 2.0 * torch.log(torch.diagonal(scale_tril, dim1=-2, dim2=-1)).sum(dim=-1)
        precision_means = torch.einsum("kij,kj->ki", precision, self.means)
        mean_precision_quad = torch.einsum("ki,ki->k", self.means, precision_means)
        return covariance, scale_tril, precision, logdet_cov, mean_precision_quad

    def endpoint_distribution(
        self,
        source_points: torch.Tensor,
        *,
        epsilon: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        covariance, scale_tril, _, _, _ = self.component_stats()
        log_mix = F.log_softmax(self.logits, dim=0)
        component_means = self.means.unsqueeze(0) + torch.einsum("kij,bj->bki", covariance, source_points)
        source_mean_term = source_points @ self.means.T
        source_cov_term = torch.einsum("bi,kij,bj->bk", source_points, covariance, source_points)
        log_weights = log_mix.unsqueeze(0) + (source_mean_term + 0.5 * source_cov_term) / epsilon
        component_probs = torch.softmax(log_weights, dim=1)
        component_scale_tril = math.sqrt(epsilon) * scale_tril
        return component_probs, component_means, component_scale_tril

    def _drift_impl(
        self,
        points: torch.Tensor,
        times: torch.Tensor,
        *,
        epsilon: float,
    ) -> torch.Tensor:
        covariance, _, precision, logdet_cov, mean_precision_quad = self.component_stats()
        log_mix = F.log_softmax(self.logits, dim=0)
        one_minus_t = (1.0 - times).clamp_min(DEFAULT_TIME_EPS)
        ratio = times / one_minus_t
        eye = torch.eye(self.dim, dtype=points.dtype, device=points.device)
        system = precision.unsqueeze(0) + ratio[:, None, None, None] * eye
        chol = stable_cholesky(system)
        precision_means = torch.einsum("kij,kj->ki", precision, self.means)
        rhs = points[:, None, :] / one_minus_t[:, None, None] + precision_means.unsqueeze(0)
        component_means = torch.cholesky_solve(rhs.unsqueeze(-1), chol).squeeze(-1)
        logdet_system = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=-1)
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

    def drift(
        self,
        points: torch.Tensor,
        times: torch.Tensor,
        *,
        epsilon: float,
        point_chunk_size: int | None = None,
    ) -> torch.Tensor:
        if point_chunk_size is None or point_chunk_size <= 0 or points.shape[0] <= point_chunk_size:
            return self._drift_impl(points, times, epsilon=epsilon)
        outputs = []
        for start in range(0, points.shape[0], point_chunk_size):
            points_chunk = points[start: start + point_chunk_size]
            times_chunk = times[start: start + point_chunk_size]
            outputs.append(self._drift_impl(points_chunk, times_chunk, epsilon=epsilon))
        return torch.cat(outputs, dim=0)

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
            raise ValueError("OT input plan requires an OT sampler.")
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
    bridge_mean = (1.0 - times)[:, None] * source_points + times[:, None] * target_points
    if sigma <= 0:
        return bridge_mean
    bridge_std = sigma * torch.sqrt(times * (1.0 - times))
    return bridge_mean + bridge_std[:, None] * torch.randn_like(source_points)


def train_pairwise_lightsb_m(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    *,
    sigma: float,
    input_plan: str,
    ot_method: str,
    batch_size: int,
    epochs: int,
    train_seconds: float | None,
    num_components: int,
    point_chunk_size: int,
    lr: float,
    weight_decay: float,
    min_cov_scale: float,
    device: torch.device,
    progress_label: str,
    quality_eval_every: int,
    checkpoint_callback: Callable[[int, GaussianMixtureAdjustedPotential], None] | None = None,
    epoch_callback: Callable[[int, GaussianMixtureAdjustedPotential], bool | None] | None = None,
) -> tuple[GaussianMixtureAdjustedPotential, int]:
    epsilon = sigma * sigma
    model = GaussianMixtureAdjustedPotential(
        dim=int(source_points.shape[1]),
        num_components=num_components,
        init_points=target_points.to(dtype=torch.float32, device="cpu"),
        min_cov_scale=min_cov_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ot_sampler = OTPlanSampler(method=ot_method, reg=2.0 * epsilon) if input_plan == "ot" else None
    _, steps_per_epoch = compute_reference_updates(
        source_points,
        target_points,
        batch_size=batch_size,
        epochs=1,
    )
    total_epochs = epochs if epochs > 0 else None
    progress = tqdm(total=total_epochs, desc=progress_label, leave=False, dynamic_ncols=True)
    running_loss = None
    epoch_idx = 0
    while total_epochs is None or epoch_idx < total_epochs:
        epoch_loss = 0.0
        n_updates = 0
        model.train()
        for _ in range(steps_per_epoch):
            x0, x1 = sample_training_plan(
                source_points,
                target_points,
                batch_size=batch_size,
                input_plan=input_plan,
                ot_sampler=ot_sampler,
                device=device,
            )
            times = torch.rand(batch_size, device=device, dtype=x0.dtype).clamp_(DEFAULT_TIME_EPS, 1.0 - DEFAULT_TIME_EPS)
            xt = sample_brownian_bridge(x0, x1, times, sigma=sigma)
            target_drift = (x1 - xt) / (1.0 - times)[:, None]
            predicted_drift = model.drift(
                xt,
                times,
                epsilon=epsilon,
                point_chunk_size=point_chunk_size,
            )
            loss = torch.mean((predicted_drift - target_drift) ** 2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            n_updates += 1

        mean_loss = epoch_loss / max(n_updates, 1)
        running_loss = mean_loss if running_loss is None else 0.95 * running_loss + 0.05 * mean_loss
        progress.update(1)
        progress.set_postfix(loss=f"{running_loss:.5f}")
        epoch_number = epoch_idx + 1
        if checkpoint_callback is not None and quality_eval_every > 0 and (
            epoch_number % quality_eval_every == 0 or epoch_number == total_epochs
        ):
            checkpoint_callback(epoch_number, model)
        should_stop = False
        if epoch_callback is not None:
            should_stop = bool(epoch_callback(epoch_number, model))
        epoch_idx += 1
        if should_stop:
            break
    progress.close()
    return model, epoch_idx


@torch.inference_mode()
def sample_predicted_target_lightsb_m(
    model: GaussianMixtureAdjustedPotential,
    source_points: torch.Tensor,
    *,
    sigma: float,
    rollout_batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    epsilon = sigma * sigma
    outputs = []
    model.eval()
    for start in range(0, source_points.shape[0], rollout_batch_size):
        batch = source_points[start: start + rollout_batch_size].to(device)
        outputs.append(model.sample_endpoint_given_source(batch, epsilon=epsilon).cpu())
    return torch.cat(outputs, dim=0)
