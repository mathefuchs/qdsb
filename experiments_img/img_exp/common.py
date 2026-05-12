from __future__ import annotations

import math
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


def sample_minibatch(points: torch.Tensor, batch_size: int) -> torch.Tensor:
    indices = torch.randint(0, points.shape[0], (batch_size,))
    return points[indices]


def subsample_points(points: torch.Tensor, max_points: int | None) -> torch.Tensor:
    if max_points is None or max_points <= 0 or points.shape[0] <= max_points:
        return points
    indices = torch.randperm(points.shape[0])[:max_points]
    return points[indices]


def compute_reference_updates(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    *,
    batch_size: int,
    epochs: int,
) -> tuple[int, int]:
    max_points = max(source_points.shape[0], target_points.shape[0])
    steps_per_epoch = max(1, math.ceil(max_points / batch_size))
    return epochs * steps_per_epoch, steps_per_epoch


def format_metric(mean: float, std: float) -> str:
    return f"{mean:.3f}\u00b1{std:.3f}"

