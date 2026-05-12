from __future__ import annotations

from dataclasses import dataclass

import torch
from torchcfm.utils import sample_8gaussians, sample_moons


@dataclass(frozen=True)
class ToyDatasetSpec:
    key: str
    label: str
    source_name: str
    target_name: str


def _sample_gaussian(n):
    return 1.5 * torch.randn((n, 2))


def sample_toy_dataset(name: str, n: int):
    """ Sample one of three datasets """

    if name == "8gaussians":
        return sample_8gaussians(n)
    if name == "gaussian":
        return _sample_gaussian(n)
    if name == "moons":
        return sample_moons(n)

    raise ValueError("Unkown dataset")


TOY_DATASETS = {
    "8gaussians_moons": ToyDatasetSpec(
        key="8gaussians_moons",
        label="8Gaussians -> Moons",
        source_name="8gaussians",
        target_name="moons",
    ),
    "gaussian_moons": ToyDatasetSpec(
        key="gaussian_moons",
        label="Gaussian -> Moons",
        source_name="gaussian",
        target_name="moons",
    ),
    "gaussian_8gaussians": ToyDatasetSpec(
        key="gaussian_8gaussians",
        label="Gaussian -> 8Gaussians",
        source_name="gaussian",
        target_name="8gaussians",
    ),
}
TOY_DATASET_ORDER = tuple(TOY_DATASETS.keys())


def sample_toy_problem(
    spec: ToyDatasetSpec,
    *,
    num_samples: int,
    num_eval_samples: int,
) -> dict[str, torch.Tensor]:
    return {
        "train_source": sample_toy_dataset(spec.source_name, num_samples).float(),
        "train_target": sample_toy_dataset(spec.target_name, num_samples).float(),
        "eval_source": sample_toy_dataset(spec.source_name, num_eval_samples).float(),
        "eval_target": sample_toy_dataset(spec.target_name, num_eval_samples).float(),
    }
