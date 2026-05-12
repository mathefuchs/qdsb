from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcfm.optimal_transport import OTPlanSampler
from tqdm import tqdm

from toy_exp.common import compute_reference_updates, sample_minibatch

DEFAULT_MIN_COV_SCALE = 1e-2
DEFAULT_TIME_EPS = 1e-4


def inverse_softplus(value: float) -> float:
    if value <= 0:
        raise ValueError("softplus inverse expects a positive value.")
    return float(math.log(math.expm1(value)))


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
        indices = torch.randint(0, init_points.shape[0], (num_components,))
        means_init = init_points[indices].clone()
        var = init_points.var(dim=0, unbiased=False)
        avg_std = float(torch.sqrt(var.mean().clamp_min(1e-6)).item())
        diag_init = max(avg_std, min_cov_scale)
        raw_diag_init = inverse_softplus(diag_init - min_cov_scale)
        raw_tril = torch.zeros(num_components, dim, dim, dtype=init_points.dtype)
        diag_indices = torch.arange(dim)
        raw_tril[:, diag_indices, diag_indices] = raw_diag_init

        self.dim = dim
        self.min_cov_scale = min_cov_scale
        self.logits = nn.Parameter(torch.zeros(num_components, dtype=init_points.dtype))
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

    def drift(
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
        chol = torch.linalg.cholesky(system)
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
    num_components: int,
    lr: float,
    weight_decay: float,
    min_cov_scale: float,
    device: torch.device,
    progress_label: str,
    quality_eval_every: int,
    checkpoint_callback: Callable[[int, GaussianMixtureAdjustedPotential], None] | None = None,
) -> GaussianMixtureAdjustedPotential:
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
        epochs=epochs,
    )
    progress = tqdm(range(epochs), desc=progress_label, leave=False, dynamic_ncols=True)
    running_loss = None
    for epoch_idx in progress:
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
            predicted_drift = model.drift(xt, times, epsilon=epsilon)
            loss = torch.mean((predicted_drift - target_drift) ** 2)

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
            checkpoint_callback(epoch_number, model)
    return model


@torch.inference_mode()
def sample_predicted_target(
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
