from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from tqdm import tqdm

ALAE_MODULE_LOAD_ORDER = ("lreq", "utils", "registry", "losses", "net", "model")
DEFAULT_MODEL_CONFIG = {
    "LATENT_SPACE_SIZE": 512,
    "LAYER_COUNT": 9,
    "MAX_CHANNEL_COUNT": 512,
    "START_CHANNEL_COUNT": 16,
    "DLATENT_AVG_BETA": 0.995,
    "TRUNCATIOM_PSI": 0.7,
    "TRUNCATIOM_CUTOFF": 8,
    "STYLE_MIXING_PROB": 0.9,
    "MAPPING_LAYERS": 8,
    "CHANNELS": 3,
    "GENERATOR": "GeneratorDefault",
    "ENCODER": "EncoderDefault",
    "Z_REGRESSION": False,
}


@dataclass(frozen=True)
class ALAEConfig:
    root: Path
    config_path: Path
    checkpoint_path: Path
    output_dir: Path
    latent_dim: int
    layer_count: int
    start_channels: int
    max_channels: int
    mapping_layers: int
    channels: int
    generator: str
    encoder: str
    dlatent_avg_beta: float | None
    truncation_psi: float | None
    truncation_cutoff: int | None
    style_mixing_prob: float | None
    z_regression: bool

    @property
    def resolution(self) -> int:
        return 2 ** (self.layer_count + 1)


def _load_python_module(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create module spec for {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_alae_modules(root: Path) -> dict[str, Any]:
    original_modules = {name: sys.modules.get(name) for name in ALAE_MODULE_LOAD_ORDER}
    loaded_modules: dict[str, Any] = {}
    try:
        for module_name in ALAE_MODULE_LOAD_ORDER:
            loaded_modules[module_name] = _load_python_module(
                module_name,
                root / f"{module_name}.py",
            )
    finally:
        for module_name, original in original_modules.items():
            if original is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original
    return loaded_modules


def _merge_model_config(payload: dict[str, Any]) -> dict[str, Any]:
    merged = dict(DEFAULT_MODEL_CONFIG)
    merged.update(payload.get("MODEL", {}))
    return merged


def resolve_alae_config_path(root: Path, config_spec: str | Path) -> Path:
    candidate = Path(config_spec)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate
    if candidate.suffix:
        config_path = root / candidate
    else:
        config_path = root / "configs" / f"{candidate}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"ALAE config not found: {config_path}")
    return config_path


def load_alae_config(
    *,
    root: Path,
    config_spec: str | Path,
    checkpoint_path: Path | None,
) -> ALAEConfig:
    config_path = resolve_alae_config_path(root, config_spec)
    payload = yaml.safe_load(config_path.read_text()) or {}
    model_cfg = _merge_model_config(payload)
    output_dir = Path(payload.get("OUTPUT_DIR", "training_artifacts/ffhq"))
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    if checkpoint_path is None:
        last_checkpoint_path = output_dir / "last_checkpoint"
        if not last_checkpoint_path.exists():
            raise FileNotFoundError(
                "ALAE checkpoint pointer not found at "
                f"{last_checkpoint_path}. Provide --alae-checkpoint explicitly."
            )
        checkpoint_ref = last_checkpoint_path.read_text().strip()
        resolved_checkpoint = Path(checkpoint_ref)
        if not resolved_checkpoint.is_absolute():
            resolved_checkpoint = root / resolved_checkpoint
    else:
        resolved_checkpoint = checkpoint_path
    if not resolved_checkpoint.exists():
        raise FileNotFoundError(
            "ALAE checkpoint not found at "
            f"{resolved_checkpoint}. The cloned ALAE repo includes the pointer, but the actual "
            "checkpoint file still needs to be present locally."
        )

    return ALAEConfig(
        root=root,
        config_path=config_path,
        checkpoint_path=resolved_checkpoint,
        output_dir=output_dir,
        latent_dim=int(model_cfg["LATENT_SPACE_SIZE"]),
        layer_count=int(model_cfg["LAYER_COUNT"]),
        start_channels=int(model_cfg["START_CHANNEL_COUNT"]),
        max_channels=int(model_cfg["MAX_CHANNEL_COUNT"]),
        mapping_layers=int(model_cfg["MAPPING_LAYERS"]),
        channels=int(model_cfg["CHANNELS"]),
        generator=str(model_cfg["GENERATOR"]),
        encoder=str(model_cfg["ENCODER"]),
        dlatent_avg_beta=(
            None if model_cfg.get("DLATENT_AVG_BETA") is None else float(model_cfg["DLATENT_AVG_BETA"])
        ),
        truncation_psi=(
            None if model_cfg.get("TRUNCATIOM_PSI") is None else float(model_cfg["TRUNCATIOM_PSI"])
        ),
        truncation_cutoff=(
            None if model_cfg.get("TRUNCATIOM_CUTOFF") is None else int(model_cfg["TRUNCATIOM_CUTOFF"])
        ),
        style_mixing_prob=(
            None if model_cfg.get("STYLE_MIXING_PROB") is None else float(model_cfg["STYLE_MIXING_PROB"])
        ),
        z_regression=bool(model_cfg.get("Z_REGRESSION", False)),
    )


def _extract_checkpoint_models(checkpoint: dict[str, Any]) -> dict[str, Any]:
    if "models" in checkpoint and isinstance(checkpoint["models"], dict):
        return checkpoint["models"]
    return checkpoint


def _load_component_state(model: torch.nn.Module, state_dict: dict[str, Any], key: str) -> None:
    if key not in state_dict:
        raise KeyError(f"Checkpoint is missing the ALAE component {key!r}.")
    model.load_state_dict(state_dict[key], strict=False)


def load_alae_checkpoint(path: Path, *, root: Path) -> dict[str, Any]:
    original_tracker = sys.modules.get("tracker")
    try:
        tracker_path = root / "tracker.py"
        if tracker_path.exists():
            _load_python_module("tracker", tracker_path)
        return torch.load(path, map_location="cpu", weights_only=False)
    finally:
        if original_tracker is None:
            sys.modules.pop("tracker", None)
        else:
            sys.modules["tracker"] = original_tracker


class ALAEInference:
    def __init__(
        self,
        *,
        root: Path,
        config_spec: str | Path = "ffhq",
        checkpoint_path: Path | None = None,
        device: torch.device,
    ) -> None:
        self.config = load_alae_config(
            root=root,
            config_spec=config_spec,
            checkpoint_path=checkpoint_path,
        )
        modules = load_alae_modules(self.config.root)
        modules["lreq"].use_implicit_lreq.set(True)
        model_cls = modules["model"].Model
        self.model = model_cls(
            startf=self.config.start_channels,
            maxf=self.config.max_channels,
            layer_count=self.config.layer_count,
            latent_size=self.config.latent_dim,
            mapping_layers=self.config.mapping_layers,
            dlatent_avg_beta=self.config.dlatent_avg_beta,
            truncation_psi=self.config.truncation_psi,
            truncation_cutoff=self.config.truncation_cutoff,
            style_mixing_prob=self.config.style_mixing_prob,
            channels=self.config.channels,
            generator=self.config.generator,
            encoder=self.config.encoder,
            z_regression=self.config.z_regression,
        ).to(device)
        checkpoint = load_alae_checkpoint(self.config.checkpoint_path, root=self.config.root)
        checkpoint_models = _extract_checkpoint_models(checkpoint)
        _load_component_state(self.model.encoder, checkpoint_models, "discriminator_s")
        _load_component_state(self.model.decoder, checkpoint_models, "generator_s")
        _load_component_state(self.model.mapping_d, checkpoint_models, "mapping_tl_s")
        _load_component_state(self.model.mapping_f, checkpoint_models, "mapping_fl_s")
        _load_component_state(self.model.dlatent_avg, checkpoint_models, "dlatent_avg")
        self.model.eval()
        self.model.requires_grad_(False)
        self.device = device
        self.lod = self.config.layer_count - 1
        self.blend_factor = 1.0
        self.num_style_layers = int(self.model.mapping_f.num_layers)

    @property
    def latent_dim(self) -> int:
        return int(self.config.latent_dim)

    @property
    def resolution(self) -> int:
        return int(self.config.resolution)

    @torch.inference_mode()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device, non_blocking=True)
        latent, _ = self.model.encode(images, self.lod, self.blend_factor)
        return latent[:, 0].detach().cpu()

    @torch.inference_mode()
    def decode_latents(
        self,
        latents: torch.Tensor,
        *,
        batch_size: int,
        noise: bool = False,
    ) -> torch.Tensor:
        outputs = []
        for start in range(0, latents.shape[0], batch_size):
            batch = latents[start: start + batch_size].to(self.device, non_blocking=True)
            styles = batch[:, None, :].repeat(1, self.num_style_layers, 1)
            decoded = self.model.decoder(styles, self.lod, self.blend_factor, noise=noise)
            outputs.append(decoded.detach().cpu())
        return torch.cat(outputs, dim=0)


def preprocess_pil_image(image: Image.Image, *, resolution: int) -> torch.Tensor:
    image = image.convert("RGB")
    tensor = torch.from_numpy(np.array(image, dtype=np.uint8, copy=True)).permute(2, 0, 1).float() / 127.5 - 1.0
    tensor = tensor.unsqueeze(0)
    while tensor.shape[-2] > resolution or tensor.shape[-1] > resolution:
        tensor = F.avg_pool2d(tensor, kernel_size=2, stride=2)
    if tensor.shape[-2:] != (resolution, resolution):
        tensor = F.adaptive_avg_pool2d(tensor, (resolution, resolution))
    return tensor.squeeze(0).contiguous()


def encode_image_paths(
    alae: ALAEInference,
    image_paths: list[Path],
    *,
    batch_size: int,
    progress_label: str | None = None,
) -> torch.Tensor:
    latents = []
    iterator = range(0, len(image_paths), batch_size)
    if progress_label is not None:
        iterator = tqdm(iterator, desc=progress_label, leave=False)
    for start in iterator:
        batch_paths = image_paths[start: start + batch_size]
        batch_tensors = []
        for path in batch_paths:
            with Image.open(path) as image:
                batch_tensors.append(preprocess_pil_image(image, resolution=alae.resolution))
        batch_images = torch.stack(batch_tensors, dim=0)
        latents.append(alae.encode_images(batch_images))
    return torch.cat(latents, dim=0)


def save_decoded_images(
    alae: ALAEInference,
    latents: torch.Tensor,
    *,
    output_dir: Path,
    prefix: str,
    batch_size: int,
    noise: bool = False,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    decoded = alae.decode_latents(
        latents,
        batch_size=batch_size,
        noise=noise,
    )
    saved_paths: list[str] = []
    for index, image_tensor in enumerate(decoded):
        image = ((image_tensor.clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)
        pil_image = Image.fromarray(image.permute(1, 2, 0).cpu().numpy(), mode="RGB")
        path = output_dir / f"{prefix}_{index:04d}.png"
        pil_image.save(path)
        saved_paths.append(str(path))
    return saved_paths
