from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model_times(times: np.ndarray, mode: str) -> np.ndarray:
    if mode == "discrete":
        return np.arange(len(times), dtype=np.float32)

    shifted = (times - times[0]).astype(np.float32, copy=False)
    if mode == "raw":
        return shifted
    if mode == "scaled":
        span = shifted[-1]
        if span <= 0:
            return np.zeros_like(times)
        return shifted / span * (len(times) - 1)
    raise ValueError(f"Unsupported time mode: {mode}")


def sample_minibatch(points: torch.Tensor, batch_size: int) -> torch.Tensor:
    indices = torch.randint(0, points.shape[0], (batch_size,))
    return points[indices]


def build_leaveout_pair_indices(
    *,
    num_timepoints: int,
    missing_index: int,
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for src_idx in range(num_timepoints - 1):
        if src_idx == missing_index:
            continue
        dst_idx = src_idx + 1
        if dst_idx == missing_index:
            dst_idx += 1
        if dst_idx >= num_timepoints:
            continue
        pairs.append((src_idx, dst_idx))
    if not pairs:
        raise ValueError(
            f"No training pairs remain after leaving out index {missing_index}."
        )
    return pairs


def subsample_points(points: torch.Tensor, max_points: int | None) -> torch.Tensor:
    if max_points is None or max_points <= 0 or points.shape[0] <= max_points:
        return points
    indices = torch.randperm(points.shape[0])[:max_points]
    return points[indices]


def find_bracketing_interval(
    observed_indices: list[int],
    missing_index: int,
) -> tuple[int, int]:
    for left, right in zip(observed_indices[:-1], observed_indices[1:]):
        if left < missing_index < right:
            return left, right
    raise ValueError(
        f"Missing index {missing_index} is not bracketed by observed timepoints."
    )


def format_metric(mean: float, std: float) -> str:
    return f"{mean:.3f}\u00b1{std:.3f}"


def resolve_dataset_sigma(
    dataset_key: str,
    *,
    sigma: float | None,
    sigma_eb: float,
    sigma_cite: float,
    sigma_multi: float,
) -> float:
    if sigma is not None:
        return sigma
    if dataset_key == "eb":
        return sigma_eb
    if dataset_key == "cite":
        return sigma_cite
    if dataset_key == "multi":
        return sigma_multi
    raise ValueError(f"Unsupported dataset key for sigma resolution: {dataset_key}")
