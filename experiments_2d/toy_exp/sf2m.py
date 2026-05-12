from __future__ import annotations

import math
import random
import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
import ot as pot
import torch
from torchcfm.conditional_flow_matching import SchrodingerBridgeConditionalFlowMatcher
from torchcfm.models import MLP
from tqdm import tqdm

from toy_exp.common import compute_reference_updates, sample_minibatch


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
def sample_predicted_target(
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
    width: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    progress_label: str,
    quality_eval_every: int,
    checkpoint_callback: Callable[[int, MLP, MLP], None] | None = None,
) -> tuple[MLP, MLP]:
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
        epochs=epochs,
    )
    progress = tqdm(range(epochs), desc=progress_label, leave=False, dynamic_ncols=True)
    running_loss = None

    for epoch_idx in progress:
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
        progress.set_postfix(loss=f"{running_loss:.5f}")

        epoch_number = epoch_idx + 1
        if checkpoint_callback is not None and quality_eval_every > 0 and (
            epoch_number % quality_eval_every == 0 or epoch_number == epochs
        ):
            checkpoint_callback(epoch_number, flow_model, score_model)

    return flow_model, score_model


class MPOTPlanSampler:
    def __init__(
        self,
        *,
        method: str,
        transport_fraction: float,
        reg: float = 0.05,
        warn: bool = True,
    ) -> None:
        if not (0.0 < transport_fraction <= 1.0):
            raise ValueError("transport_fraction must lie in (0, 1].")
        if method not in {"exact", "sinkhorn"}:
            raise ValueError(f"Unsupported m-POT method {method!r}.")
        self.method = method
        self.transport_fraction = transport_fraction
        self.reg = reg
        self.warn = warn

    def get_map(self, x0: torch.Tensor, x1: torch.Tensor) -> np.ndarray:
        a = pot.unif(x0.shape[0])
        b = pot.unif(x1.shape[0])
        cost = torch.cdist(x0, x1) ** 2
        cost_np = cost.detach().cpu().numpy()
        if self.method == "exact":
            plan = pot.partial.partial_wasserstein(
                a,
                b,
                cost_np,
                m=self.transport_fraction,
                nb_dummies=1,
            )
        else:
            plan = pot.partial.entropic_partial_wasserstein(
                a,
                b,
                cost_np,
                reg=self.reg,
                m=self.transport_fraction,
            )
        if not np.all(np.isfinite(plan)) or float(plan.sum()) <= 1e-12:
            if self.warn:
                warnings.warn(
                    "Degenerate m-POT plan, reverting to uniform plan.",
                    stacklevel=2,
                )
            plan = np.ones((x0.shape[0], x1.shape[0]), dtype=np.float64)
        return plan / float(plan.sum())

    def sample_plan(self, x0: torch.Tensor, x1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        plan = self.get_map(x0, x1)
        choices = np.random.choice(
            plan.shape[0] * plan.shape[1],
            p=plan.reshape(-1),
            size=x0.shape[0],
            replace=True,
        )
        i, j = np.divmod(choices, plan.shape[1])
        return x0[i], x1[j]


class SchrodingerBridgeMPOTConditionalFlowMatcher(SchrodingerBridgeConditionalFlowMatcher):
    def __init__(
        self,
        *,
        sigma: float,
        ot_method: str,
        transport_fraction: float,
    ) -> None:
        super().__init__(sigma=sigma, ot_method=ot_method)
        self.ot_sampler = MPOTPlanSampler(
            method=ot_method,
            transport_fraction=transport_fraction,
            reg=2 * self.sigma**2,
        )


def train_pairwise_sf2m_mpot(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    *,
    sigma: float,
    ot_method: str,
    mpot_fraction: float,
    batch_size: int,
    epochs: int,
    width: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    progress_label: str,
    quality_eval_every: int,
    checkpoint_callback: Callable[[int, MLP, MLP], None] | None = None,
) -> tuple[MLP, MLP]:
    dim = int(source_points.shape[1])
    flow_model, score_model = build_models(dim=dim, width=width, device=device)
    optimizer = torch.optim.AdamW(
        list(flow_model.parameters()) + list(score_model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    flow_matcher = SchrodingerBridgeMPOTConditionalFlowMatcher(
        sigma=sigma,
        ot_method=ot_method,
        transport_fraction=mpot_fraction,
    )
    _, steps_per_epoch = compute_reference_updates(
        source_points,
        target_points,
        batch_size=batch_size,
        epochs=epochs,
    )
    progress = tqdm(range(epochs), desc=progress_label, leave=False, dynamic_ncols=True)
    running_loss = None

    for epoch_idx in progress:
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
        progress.set_postfix(loss=f"{running_loss:.5f}")

        epoch_number = epoch_idx + 1
        if checkpoint_callback is not None and quality_eval_every > 0 and (
            epoch_number % quality_eval_every == 0 or epoch_number == epochs
        ):
            checkpoint_callback(epoch_number, flow_model, score_model)

    return flow_model, score_model


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
    width: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    progress_label: str,
    quality_eval_every: int,
    checkpoint_callback: Callable[[int, MLP, MLP], None] | None = None,
    coverage_refresh_callback: Callable[[int, CoverageAccelerationPlan], None] | None = None,
) -> tuple[MLP, MLP]:
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
    if coverage_refresh_callback is not None:
        coverage_refresh_callback(0, coverage_plan)
    _, steps_per_epoch = compute_reference_updates(
        source_points,
        target_points,
        batch_size=batch_size,
        epochs=epochs,
    )
    progress = tqdm(range(epochs), desc=progress_label, leave=False, dynamic_ncols=True)
    running_loss = None

    for epoch_idx in progress:
        if (
            coverage_anchor_refresh_epochs > 0
            and epoch_idx > 0
            and epoch_idx % coverage_anchor_refresh_epochs == 0
        ):
            coverage_plan = refresh_coverage_plan()
            if coverage_refresh_callback is not None:
                coverage_refresh_callback(epoch_idx + 1, coverage_plan)

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
        progress.set_postfix(loss=f"{running_loss:.5f}")

        epoch_number = epoch_idx + 1
        if checkpoint_callback is not None and quality_eval_every > 0 and (
            epoch_number % quality_eval_every == 0 or epoch_number == epochs
        ):
            checkpoint_callback(epoch_number, flow_model, score_model)

    return flow_model, score_model
