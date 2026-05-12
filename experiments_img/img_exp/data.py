from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from img_exp.alae import ALAEInference, encode_image_paths


@dataclass(frozen=True)
class ImageTranslationDataset:
    train_source: torch.Tensor
    train_target: torch.Tensor
    eval_source: torch.Tensor
    eval_target: torch.Tensor
    eval_source_paths: tuple[Path, ...] | None
    latent_dim: int
    source_label: str
    target_label: str
    artifact_desc: str


REQUIRED_KEYS = {
    "train_source": ("train_source", "source_train", "src_train"),
    "train_target": ("train_target", "target_train", "tgt_train"),
    "eval_source": ("eval_source", "source_eval", "src_eval", "test_source", "source_test"),
    "eval_target": ("eval_target", "target_eval", "tgt_eval", "test_target", "target_test"),
}
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _coerce_tensor(value: Any, *, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().float()
    else:
        tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must have shape [num_samples, latent_dim], got {tuple(tensor.shape)}.")
    if tensor.shape[0] == 0:
        raise ValueError(f"{name} must not be empty.")
    return tensor.contiguous()


def _load_payload(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, torch.Tensor):
            raise ValueError(
                f"{path} contains a bare tensor. Use separate latent-file arguments for that format."
            )
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain a dict-like payload.")
        return dict(payload)
    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as npz:
            return {key: npz[key] for key in npz.files}
    raise ValueError(f"Unsupported bundle format: {path}")


def _resolve_key(payload: dict[str, Any], canonical_key: str) -> Any:
    for key in REQUIRED_KEYS[canonical_key]:
        if key in payload:
            return payload[key]
    raise KeyError(f"Missing key for {canonical_key!r}. Expected one of {REQUIRED_KEYS[canonical_key]}.")


def load_latent_bundle(
    path: Path,
    *,
    source_label: str,
    target_label: str,
) -> ImageTranslationDataset:
    payload = _load_payload(path)
    train_source = _coerce_tensor(_resolve_key(payload, "train_source"), name="train_source")
    train_target = _coerce_tensor(_resolve_key(payload, "train_target"), name="train_target")
    eval_source = _coerce_tensor(_resolve_key(payload, "eval_source"), name="eval_source")
    eval_target = _coerce_tensor(_resolve_key(payload, "eval_target"), name="eval_target")
    latent_dim = int(train_source.shape[1])
    for name, tensor in (
        ("train_target", train_target),
        ("eval_source", eval_source),
        ("eval_target", eval_target),
    ):
        if int(tensor.shape[1]) != latent_dim:
            raise ValueError(f"{name} latent dimension mismatch: expected {latent_dim}, got {tensor.shape[1]}.")
    return ImageTranslationDataset(
        train_source=train_source,
        train_target=train_target,
        eval_source=eval_source,
        eval_target=eval_target,
        eval_source_paths=None,
        latent_dim=latent_dim,
        source_label=source_label,
        target_label=target_label,
        artifact_desc=str(path),
    )


def _load_single_latent_file(path: Path) -> torch.Tensor:
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            if len(payload) != 1:
                raise ValueError(f"{path} contains multiple keys. Use --latents with a bundle file instead.")
            payload = next(iter(payload.values()))
        return _coerce_tensor(payload, name=str(path))
    if suffix == ".npy":
        return _coerce_tensor(np.load(path, allow_pickle=False), name=str(path))
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as npz:
            if len(npz.files) != 1:
                raise ValueError(f"{path} contains multiple arrays. Use --latents with a bundle file instead.")
            return _coerce_tensor(npz[npz.files[0]], name=str(path))
    raise ValueError(f"Unsupported latent file format: {path}")


def load_latent_files(
    *,
    train_source: Path,
    train_target: Path,
    eval_source: Path,
    eval_target: Path,
    source_label: str,
    target_label: str,
) -> ImageTranslationDataset:
    train_source_tensor = _load_single_latent_file(train_source)
    train_target_tensor = _load_single_latent_file(train_target)
    eval_source_tensor = _load_single_latent_file(eval_source)
    eval_target_tensor = _load_single_latent_file(eval_target)
    latent_dim = int(train_source_tensor.shape[1])
    for name, tensor in (
        ("train_target", train_target_tensor),
        ("eval_source", eval_source_tensor),
        ("eval_target", eval_target_tensor),
    ):
        if int(tensor.shape[1]) != latent_dim:
            raise ValueError(f"{name} latent dimension mismatch: expected {latent_dim}, got {tensor.shape[1]}.")
    artifact_desc = (
        f"train_source={train_source}, train_target={train_target}, "
        f"eval_source={eval_source}, eval_target={eval_target}"
    )
    return ImageTranslationDataset(
        train_source=train_source_tensor,
        train_target=train_target_tensor,
        eval_source=eval_source_tensor,
        eval_target=eval_target_tensor,
        eval_source_paths=None,
        latent_dim=latent_dim,
        source_label=source_label,
        target_label=target_label,
        artifact_desc=artifact_desc,
    )


def list_image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    files = sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        raise ValueError(f"No image files found in {directory}")
    return files


def load_image_directories(
    *,
    alae: ALAEInference,
    source_train_dir: Path,
    target_train_dir: Path,
    source_eval_dir: Path | None,
    target_eval_dir: Path | None,
    source_label: str,
    target_label: str,
    encode_batch_size: int,
) -> ImageTranslationDataset:
    resolved_source_eval_dir = source_train_dir if source_eval_dir is None else source_eval_dir
    resolved_target_eval_dir = target_train_dir if target_eval_dir is None else target_eval_dir
    source_train_paths = list_image_files(source_train_dir)
    target_train_paths = list_image_files(target_train_dir)
    source_eval_paths = list_image_files(resolved_source_eval_dir)
    target_eval_paths = list_image_files(resolved_target_eval_dir)

    train_source = encode_image_paths(
        alae,
        source_train_paths,
        batch_size=encode_batch_size,
        progress_label=f"Encoding {source_label} train",
    )
    train_target = encode_image_paths(
        alae,
        target_train_paths,
        batch_size=encode_batch_size,
        progress_label=f"Encoding {target_label} train",
    )
    eval_source = encode_image_paths(
        alae,
        source_eval_paths,
        batch_size=encode_batch_size,
        progress_label=f"Encoding {source_label} eval",
    )
    eval_target = encode_image_paths(
        alae,
        target_eval_paths,
        batch_size=encode_batch_size,
        progress_label=f"Encoding {target_label} eval",
    )

    artifact_desc = (
        f"raw_images: source_train={source_train_dir}, target_train={target_train_dir}, "
        f"source_eval={resolved_source_eval_dir}, target_eval={resolved_target_eval_dir}; "
        f"alae_config={alae.config.config_path}, alae_checkpoint={alae.config.checkpoint_path}"
    )
    return ImageTranslationDataset(
        train_source=train_source,
        train_target=train_target,
        eval_source=eval_source,
        eval_target=eval_target,
        eval_source_paths=tuple(source_eval_paths),
        latent_dim=alae.latent_dim,
        source_label=source_label,
        target_label=target_label,
        artifact_desc=artifact_desc,
    )
