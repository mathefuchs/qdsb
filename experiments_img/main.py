from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from img_exp.alae import ALAEInference, save_decoded_images
from img_exp.common import format_metric, resolve_device, set_seed
from img_exp.data import load_image_directories, load_latent_bundle, load_latent_files
from img_exp.dsb import DSBHyperparams, sample_predicted_target as sample_dsb_target
from img_exp.dsb import train_pairwise_dsb
from img_exp.dsbm import DSBMHyperparams, sample_predicted_target as sample_dsbm_target
from img_exp.dsbm import train_pairwise_dsbm
from img_exp.ffhq import (
    DEFAULT_FFHQ_ADULT_AGE_GROUPS,
    DEFAULT_FFHQ_CHILD_AGE_GROUPS,
    DEFAULT_FFHQ_EVAL_FRACTION,
    DEFAULT_FFHQ_ROOT,
    DEFAULT_FFHQ_SPLIT_SEED,
    ensure_prepared_ffhq_dirs,
)
from img_exp.lightsb_m import (
    DEFAULT_LIGHTSB_M_POINT_CHUNK_SIZE,
    DEFAULT_MIN_COV_SCALE,
    sample_predicted_target as sample_lightsb_target,
    train_pairwise_lightsb_m,
)
from img_exp.quality import (
    DEFAULT_QUALITY_CURVE_EVAL_POINTS,
    DEFAULT_QUALITY_EVAL_EVERY,
    QUALITY_CURVE_METRIC,
    QualityCheckpointRecorder,
    compute_mmd,
    compute_w1,
    estimate_mmd_gamma,
    summarize_quality_curve,
)
from img_exp.qdsb import sample_predicted_target as sample_qdsb_target
from img_exp.qdsb import train_pairwise_qdsb
from img_exp.sf2m import sample_predicted_target as sample_sf2m_target
from img_exp.sf2m import train_pairwise_sf2m

DEFAULT_ALGORITHM_ORDER = ("qdsb", "sf2m", "dsb", "dsbm", "lightsb_m")
ALGORITHM_LABELS = {
    "sf2m": "SF2M",
    "qdsb": "QDSB (ours)",
    "dsb": "DSB",
    "dsbm": "DSBM",
    "lightsb_m": "LightSB-M",
}
DEFAULT_COVERAGE_ANCHORS = 512
DEFAULT_COVERAGE_ANCHOR_REFRESH_EPOCHS = 0
DEFAULT_LIGHTSB_M_NUM_COMPONENTS = 10
DEFAULT_SAVE_SAMPLE_IMAGES_DIR = Path("results/output_img_samples")
DEFAULT_NUM_SAVED_IMAGES = 50
DEFAULT_TRAIN_SECONDS = 1000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run latent-space unpaired FFHQ image translation experiments using "
            "official ALAE-style latent artifacts across multiple bridge baselines."
        ),
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=list(DEFAULT_ALGORITHM_ORDER),
        choices=list(DEFAULT_ALGORITHM_ORDER),
        help="Which algorithms to run.",
    )
    parser.add_argument(
        "--latents",
        type=Path,
        default=None,
        help=(
            "Bundle file (.pt/.pth/.npz) with train_source/train_target/eval_source/eval_target "
            "latents from a pretrained ALAE encoder."
        ),
    )
    parser.add_argument("--train-source-latents", type=Path, default=None)
    parser.add_argument("--train-target-latents", type=Path, default=None)
    parser.add_argument("--eval-source-latents", type=Path, default=None)
    parser.add_argument("--eval-target-latents", type=Path, default=None)
    parser.add_argument("--source-train-dir", type=Path, default=None)
    parser.add_argument("--target-train-dir", type=Path, default=None)
    parser.add_argument("--source-eval-dir", type=Path, default=None)
    parser.add_argument("--target-eval-dir", type=Path, default=None)
    parser.add_argument("--ffhq-root", type=Path, default=DEFAULT_FFHQ_ROOT)
    parser.add_argument(
        "--ffhq-child-age-groups",
        nargs="+",
        default=list(DEFAULT_FFHQ_CHILD_AGE_GROUPS),
    )
    parser.add_argument(
        "--ffhq-adult-age-groups",
        nargs="+",
        default=list(DEFAULT_FFHQ_ADULT_AGE_GROUPS),
    )
    parser.add_argument("--ffhq-eval-fraction", type=float,
                        default=DEFAULT_FFHQ_EVAL_FRACTION)
    parser.add_argument("--ffhq-split-seed", type=int,
                        default=DEFAULT_FFHQ_SPLIT_SEED)
    parser.add_argument("--alae-root", type=Path, default=Path("ALAE"))
    parser.add_argument(
        "--alae-config",
        type=str,
        default="ffhq",
        help="ALAE config name under ALAE/configs or an explicit .yaml path.",
    )
    parser.add_argument(
        "--alae-checkpoint",
        type=Path,
        default=Path("ALAE/training_artifacts/ffhq/model_157.pth"),
    )
    parser.add_argument("--alae-encode-batch-size", type=int, default=4)
    parser.add_argument("--source-label", type=str, default="adult")
    parser.add_argument("--target-label", type=str, default="child")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--ot-method", type=str,
                        default="exact", choices=["exact", "sinkhorn"])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--train-seconds",
        type=float,
        default=DEFAULT_TRAIN_SECONDS,
        help="Train each algorithm for this many timed seconds by default.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="Optional hard cap on training epochs. Use 0 to disable the epoch cap.",
    )
    parser.add_argument(
        "--quality-eval-every",
        type=int,
        default=DEFAULT_QUALITY_EVAL_EVERY,
        help="Record untimed latent-space MMD every N epochs. Use 0 to disable.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Hidden width for the QDSB flow/score MLPs.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--steps-per-unit", type=int, default=100)
    parser.add_argument("--rollout-batch-size", type=int, default=2048)
    parser.add_argument(
        "--final-eval",
        action="store_true",
        help="Compute final latent-space W1/MMD on the evaluation populations. Disabled by default for qualitative runs.",
    )
    parser.add_argument(
        "--max-eval-points",
        type=int,
        default=512,
        help="Subsample cap for final latent-space W1/MMD evaluation. Use 0 for full populations.",
    )
    parser.add_argument("--w1-method", type=str,
                        default="exact", choices=["exact", "sinkhorn"])
    parser.add_argument("--w1-reg", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--save-sample-latents-dir", type=Path, default=None)
    parser.add_argument("--num-saved-samples", type=int, default=64)
    parser.add_argument("--save-sample-images-dir", type=Path,
                        default=DEFAULT_SAVE_SAMPLE_IMAGES_DIR)
    parser.add_argument("--num-saved-images", type=int,
                        default=DEFAULT_NUM_SAVED_IMAGES)

    parser.add_argument("--coverage-anchors", type=int,
                        default=DEFAULT_COVERAGE_ANCHORS)
    parser.add_argument(
        "--coverage-anchor-selection",
        type=str,
        default="gon_plus",
        choices=["gon", "gon_plus"],
    )
    parser.add_argument(
        "--coverage-anchor-gon-plus-candidates", type=int, default=10)
    parser.add_argument(
        "--coverage-anchor-weight-temperature", type=float, default=1.0)
    parser.add_argument(
        "--coverage-anchor-refresh-epochs",
        type=int,
        default=DEFAULT_COVERAGE_ANCHOR_REFRESH_EPOCHS,
    )
    parser.add_argument(
        "--coverage-point-match-mode",
        type=str,
        default="random",
        choices=["random", "ot"],
    )

    parser.add_argument("--lightsb-m-num-components", type=int,
                        default=DEFAULT_LIGHTSB_M_NUM_COMPONENTS)
    parser.add_argument(
        "--lightsb-m-point-chunk-size",
        type=int,
        default=DEFAULT_LIGHTSB_M_POINT_CHUNK_SIZE,
        help="Chunk size for LightSB-M drift solves to keep GPU memory bounded.",
    )
    parser.add_argument(
        "--lightsb-m-input-plan",
        type=str,
        default="independent",
        choices=["ot", "independent"],
    )
    parser.add_argument("--lightsb-m-min-cov-scale",
                        type=float, default=DEFAULT_MIN_COV_SCALE)
    parser.add_argument("--dsb-num-iter", type=int, default=50)
    parser.add_argument("--dsb-n-ipf", type=int, default=20)
    parser.add_argument("--dsb-num-steps", type=int, default=20)
    parser.add_argument("--dsb-gamma-min", type=float, default=0.01)
    parser.add_argument("--dsb-gamma-max", type=float, default=0.01)
    parser.add_argument(
        "--dsb-gamma-space",
        type=str,
        default="linspace",
        choices=["linspace", "geomspace"],
    )
    parser.add_argument("--dsb-cache-batch-size", type=int, default=512)
    parser.add_argument("--dsb-num-cache-batches", type=int, default=4)
    parser.add_argument("--dsb-time-embed-dim", type=int, default=32)
    parser.add_argument("--dsb-num-workers", type=int, default=0)

    parser.add_argument("--dsbm-num-iter", type=int, default=50)
    parser.add_argument("--dsbm-n-outer", type=int, default=20)
    parser.add_argument("--dsbm-cache-batch-size", type=int, default=512)
    parser.add_argument("--dsbm-num-cache-batches", type=int, default=4)
    parser.add_argument("--dsbm-time-embed-dim", type=int, default=32)
    parser.add_argument("--dsbm-num-workers", type=int, default=0)
    parser.add_argument("--dsbm-loss-weighting", action="store_true", default=True)
    parser.add_argument("--no-dsbm-loss-weighting", dest="dsbm_loss_weighting", action="store_false")
    return parser.parse_args()


def using_latent_inputs(args: argparse.Namespace) -> bool:
    return args.latents is not None or any(
        path is not None
        for path in (
            args.train_source_latents,
            args.train_target_latents,
            args.eval_source_latents,
            args.eval_target_latents,
        )
    )


def using_image_inputs(args: argparse.Namespace) -> bool:
    return any(
        path is not None
        for path in (
            args.source_train_dir,
            args.target_train_dir,
            args.source_eval_dir,
            args.target_eval_dir,
        )
    )


def use_default_ffhq_image_inputs(args: argparse.Namespace) -> bool:
    return not using_latent_inputs(args) and not using_image_inputs(args)


def validate_args(args: argparse.Namespace) -> None:
    latent_mode = using_latent_inputs(args)
    image_mode = using_image_inputs(args)
    if args.train_seconds <= 0:
        raise ValueError("--train-seconds must be positive.")
    if args.epochs < 0:
        raise ValueError("--epochs must be non-negative.")
    if latent_mode and image_mode:
        raise ValueError(
            "Choose either latent inputs or raw image directories, not both.")
    if latent_mode and args.latents is None:
        required = (
            args.train_source_latents,
            args.train_target_latents,
            args.eval_source_latents,
            args.eval_target_latents,
        )
        if any(path is None for path in required):
            raise ValueError(
                "Provide either --latents or all of "
                "--train-source-latents/--train-target-latents/--eval-source-latents/--eval-target-latents."
            )
    if image_mode and (args.source_train_dir is None or args.target_train_dir is None):
        raise ValueError(
            "Raw image mode requires --source-train-dir and --target-train-dir.")


def load_dataset(
    args: argparse.Namespace,
    *,
    device: torch.device,
):
    if using_latent_inputs(args):
        alae = None
        if args.latents is not None:
            dataset = load_latent_bundle(
                args.latents,
                source_label=args.source_label,
                target_label=args.target_label,
            )
        else:
            dataset = load_latent_files(
                train_source=args.train_source_latents,
                train_target=args.train_target_latents,
                eval_source=args.eval_source_latents,
                eval_target=args.eval_target_latents,
                source_label=args.source_label,
                target_label=args.target_label,
            )
        return dataset, alae
    if use_default_ffhq_image_inputs(args):
        prepared_dirs = ensure_prepared_ffhq_dirs(
            root=args.ffhq_root,
            child_age_groups=tuple(args.ffhq_child_age_groups),
            adult_age_groups=tuple(args.ffhq_adult_age_groups),
            eval_fraction=args.ffhq_eval_fraction,
            split_seed=args.ffhq_split_seed,
        )
        source_train_dir = prepared_dirs.source_train
        target_train_dir = prepared_dirs.target_train
        source_eval_dir = prepared_dirs.source_eval
        target_eval_dir = prepared_dirs.target_eval
    else:
        source_train_dir = args.source_train_dir
        target_train_dir = args.target_train_dir
        source_eval_dir = args.source_eval_dir
        target_eval_dir = args.target_eval_dir

    alae = ALAEInference(
        root=args.alae_root,
        config_spec=args.alae_config,
        checkpoint_path=args.alae_checkpoint,
        device=device,
    )
    dataset = load_image_directories(
        alae=alae,
        source_train_dir=source_train_dir,
        target_train_dir=target_train_dir,
        source_eval_dir=source_eval_dir,
        target_eval_dir=target_eval_dir,
        source_label=args.source_label,
        target_label=args.target_label,
        encode_batch_size=args.alae_encode_batch_size,
    )
    return dataset, alae


def maybe_save_predicted_latents(
    predicted: torch.Tensor,
    *,
    algorithm_key: str,
    seed: int,
    snapshot_tag: str,
    output_dir: Path | None,
    num_samples: int,
) -> str | None:
    if output_dir is None:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = predicted[: min(num_samples, predicted.shape[0])].cpu()
    path = output_dir / \
        f"{algorithm_key}_seed{seed}_{snapshot_tag}_predicted_latents.pt"
    torch.save(sample, path)
    return str(path)


def maybe_save_predicted_images(
    predicted: torch.Tensor,
    *,
    alae: ALAEInference | None,
    algorithm_key: str,
    seed: int,
    snapshot_tag: str,
    output_dir: Path | None,
    num_images: int,
    batch_size: int,
) -> list[str] | None:
    if output_dir is None or alae is None:
        return None
    subset = predicted[: min(num_images, predicted.shape[0])].cpu()
    return save_decoded_images(
        alae,
        subset,
        output_dir=output_dir,
        prefix=f"{algorithm_key}_seed{seed}_{snapshot_tag}",
        batch_size=batch_size,
        noise=False,
    )


def sample_saved_source_points(
    source_points: torch.Tensor,
    *,
    max_points: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if max_points <= 0 or source_points.shape[0] <= max_points:
        return source_points, torch.arange(source_points.shape[0], dtype=torch.long)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randperm(source_points.shape[0], generator=generator)[
        :max_points]
    return source_points[indices], indices


def maybe_save_source_images(
    *,
    eval_source_paths: tuple[Path, ...] | None,
    sampled_source_indices: torch.Tensor | None,
    algorithm_key: str,
    seed: int,
    snapshot_tag: str,
    output_dir: Path | None,
    num_images: int,
) -> list[str] | None:
    if output_dir is None or eval_source_paths is None or sampled_source_indices is None:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    capped_indices = sampled_source_indices[: min(num_images, sampled_source_indices.shape[0])]
    saved_paths: list[str] = []
    for output_index, source_index in enumerate(capped_indices.tolist()):
        source_path = eval_source_paths[source_index]
        destination = output_dir / (
            f"{algorithm_key}_seed{seed}_{snapshot_tag}_source_{output_index:04d}{source_path.suffix.lower()}"
        )
        shutil.copy2(source_path, destination)
        saved_paths.append(str(destination))
    return saved_paths


def snapshot_enabled(args: argparse.Namespace) -> bool:
    return args.save_sample_images_dir is not None or args.save_sample_latents_dir is not None


def run_single_seed(
    algorithm_key: str,
    *,
    train_source: torch.Tensor,
    train_target: torch.Tensor,
    eval_source: torch.Tensor,
    eval_target: torch.Tensor,
    eval_source_paths: tuple[Path, ...] | None,
    alae: ALAEInference | None,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    quality_mmd_gamma = None
    if args.quality_eval_every > 0 or args.final_eval:
        quality_mmd_gamma = estimate_mmd_gamma(eval_target)
    checkpoint_recorder = QualityCheckpointRecorder()

    max_snapshot_points = 0
    if snapshot_enabled(args):
        if args.save_sample_latents_dir is not None:
            max_snapshot_points = max(max_snapshot_points, args.num_saved_samples)
        if args.save_sample_images_dir is not None:
            max_snapshot_points = max(max_snapshot_points, args.num_saved_images)
    snapshot_source = None
    snapshot_source_indices = None
    if max_snapshot_points > 0:
        snapshot_source, snapshot_source_indices = sample_saved_source_points(
            eval_source,
            max_points=max_snapshot_points,
            seed=seed,
        )

    def checkpoint_mmd(predicted: torch.Tensor, *, epoch: int) -> None:
        if quality_mmd_gamma is None:
            raise RuntimeError("MMD checkpointing requested without a gamma estimate.")
        checkpoint_recorder.time_evaluation(
            epoch=epoch,
            evaluate=lambda: compute_mmd(
                predicted,
                eval_target,
                gamma=quality_mmd_gamma,
                max_points=DEFAULT_QUALITY_CURVE_EVAL_POINTS,
            ),
        )

    def save_final_outputs(
        predicted: torch.Tensor,
        *,
        epoch: int,
        elapsed_seconds: float,
    ) -> dict[str, object]:
        return {
            "tag": "final",
            "epoch": int(epoch),
            "elapsed_seconds": float(elapsed_seconds),
            "saved_predicted_latents": maybe_save_predicted_latents(
                predicted,
                algorithm_key=algorithm_key,
                seed=seed,
                snapshot_tag="final",
                output_dir=args.save_sample_latents_dir,
                num_samples=args.num_saved_samples,
            ),
            "saved_predicted_images": maybe_save_predicted_images(
                predicted,
                alae=alae,
                algorithm_key=algorithm_key,
                seed=seed,
                snapshot_tag="final",
                output_dir=args.save_sample_images_dir,
                num_images=args.num_saved_images,
                batch_size=args.alae_encode_batch_size,
            ),
            "saved_source_images": maybe_save_source_images(
                eval_source_paths=eval_source_paths,
                sampled_source_indices=snapshot_source_indices,
                algorithm_key=algorithm_key,
                seed=seed,
                snapshot_tag="final",
                output_dir=args.save_sample_images_dir,
                num_images=args.num_saved_images,
            ),
        }

    set_seed(seed)
    predicted = None
    completed_epochs = 0
    if algorithm_key == "sf2m":

        def predict_output(flow_model, score_model, source_points: torch.Tensor) -> torch.Tensor:
            return sample_sf2m_target(
                flow_model,
                score_model,
                source_points,
                sigma=args.sigma,
                steps_per_unit=args.steps_per_unit,
                rollout_batch_size=args.rollout_batch_size,
                device=device,
            )

        def callback(epoch: int, flow_model, score_model) -> bool:
            if args.quality_eval_every > 0:
                checkpoint_mmd(predict_output(flow_model, score_model, eval_source), epoch=epoch)
            return checkpoint_recorder.elapsed_seconds() >= args.train_seconds

        flow_model, score_model, completed_epochs = train_pairwise_sf2m(
            train_source,
            train_target,
            sigma=args.sigma,
            ot_method=args.ot_method,
            batch_size=args.batch_size,
            epochs=args.epochs,
            train_seconds=args.train_seconds,
            width=args.width,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            progress_label=f"{ALGORITHM_LABELS[algorithm_key]} seed={seed}",
            quality_eval_every=args.quality_eval_every,
            checkpoint_callback=None,
            epoch_callback=callback,
        )
        elapsed_seconds = checkpoint_recorder.elapsed_seconds()
        if args.final_eval:
            predicted = predict_output(flow_model, score_model, eval_source)
        final_prediction_source = snapshot_source if snapshot_source is not None else eval_source
        final_snapshot = checkpoint_recorder.time_block(
            lambda: save_final_outputs(
                predict_output(flow_model, score_model, final_prediction_source),
                epoch=completed_epochs,
                elapsed_seconds=elapsed_seconds,
            )
        )
    elif algorithm_key == "qdsb":

        def predict_output(flow_model, score_model, source_points: torch.Tensor) -> torch.Tensor:
            return sample_qdsb_target(
                flow_model,
                score_model,
                source_points,
                sigma=args.sigma,
                steps_per_unit=args.steps_per_unit,
                rollout_batch_size=args.rollout_batch_size,
                device=device,
            )

        def callback(epoch: int, flow_model, score_model) -> bool:
            if args.quality_eval_every > 0:
                checkpoint_mmd(predict_output(flow_model, score_model, eval_source), epoch=epoch)
            return checkpoint_recorder.elapsed_seconds() >= args.train_seconds

        flow_model, score_model, completed_epochs = train_pairwise_qdsb(
            train_source,
            train_target,
            coverage_anchors=args.coverage_anchors,
            coverage_anchor_selection=args.coverage_anchor_selection,
            coverage_anchor_gon_plus_candidates=args.coverage_anchor_gon_plus_candidates,
            coverage_anchor_weight_temperature=args.coverage_anchor_weight_temperature,
            coverage_anchor_refresh_epochs=args.coverage_anchor_refresh_epochs,
            coverage_point_match_mode=args.coverage_point_match_mode,
            sigma=args.sigma,
            ot_method=args.ot_method,
            batch_size=args.batch_size,
            epochs=args.epochs,
            train_seconds=args.train_seconds,
            width=args.width,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            progress_label=f"{ALGORITHM_LABELS[algorithm_key]} seed={seed}",
            quality_eval_every=args.quality_eval_every,
            checkpoint_callback=None,
            epoch_callback=callback,
        )
        elapsed_seconds = checkpoint_recorder.elapsed_seconds()
        if args.final_eval:
            predicted = predict_output(flow_model, score_model, eval_source)
        final_prediction_source = snapshot_source if snapshot_source is not None else eval_source
        final_snapshot = checkpoint_recorder.time_block(
            lambda: save_final_outputs(
                predict_output(flow_model, score_model, final_prediction_source),
                epoch=completed_epochs,
                elapsed_seconds=elapsed_seconds,
            )
        )
    elif algorithm_key == "dsb":

        def dsb_checkpoint_callback(epoch: int, trainer) -> None:
            checkpoint_mmd(
                sample_dsb_target(
                    trainer,
                    eval_source,
                    rollout_batch_size=args.rollout_batch_size,
                ),
                epoch=epoch,
            )

        hyperparams = DSBHyperparams(
            batch_size=args.batch_size,
            cache_batch_size=args.dsb_cache_batch_size,
            num_cache_batches=args.dsb_num_cache_batches,
            num_iter=args.dsb_num_iter,
            n_ipf=args.dsb_n_ipf,
            lr=args.lr,
            num_steps=args.dsb_num_steps,
            gamma_min=args.dsb_gamma_min,
            gamma_max=args.dsb_gamma_max,
            gamma_space=args.dsb_gamma_space,
            hidden_dim=args.width,
            time_embed_dim=args.dsb_time_embed_dim,
            num_workers=args.dsb_num_workers,
        )
        trainer, completed_epochs = train_pairwise_dsb(
            train_source,
            train_target,
            hyperparams=hyperparams,
            epochs=args.epochs,
            train_seconds=args.train_seconds,
            device=device,
            progress_label=f"{ALGORITHM_LABELS[algorithm_key]} seed={seed}",
            quality_eval_every=args.quality_eval_every,
            checkpoint_callback=dsb_checkpoint_callback if args.quality_eval_every > 0 else None,
            stop_callback=lambda _updates: checkpoint_recorder.elapsed_seconds() >= args.train_seconds,
        )
        elapsed_seconds = checkpoint_recorder.elapsed_seconds()
        if args.final_eval:
            predicted = sample_dsb_target(
                trainer,
                eval_source,
                rollout_batch_size=args.rollout_batch_size,
            )
        final_prediction_source = snapshot_source if snapshot_source is not None else eval_source
        final_snapshot = checkpoint_recorder.time_block(
            lambda: save_final_outputs(
                sample_dsb_target(
                    trainer,
                    final_prediction_source,
                    rollout_batch_size=args.rollout_batch_size,
                ),
                epoch=completed_epochs,
                elapsed_seconds=elapsed_seconds,
            )
        )
    elif algorithm_key == "dsbm":

        def dsbm_checkpoint_callback(epoch: int, trainer) -> None:
            checkpoint_mmd(
                sample_dsbm_target(
                    trainer,
                    eval_source,
                    rollout_batch_size=args.rollout_batch_size,
                ),
                epoch=epoch,
            )

        hyperparams = DSBMHyperparams(
            batch_size=args.batch_size,
            cache_batch_size=args.dsbm_cache_batch_size,
            num_cache_batches=args.dsbm_num_cache_batches,
            num_iter=args.dsbm_num_iter,
            n_outer=args.dsbm_n_outer,
            lr=args.lr,
            sigma=args.sigma,
            steps_per_unit=args.steps_per_unit,
            hidden_dim=args.width,
            time_embed_dim=args.dsbm_time_embed_dim,
            num_workers=args.dsbm_num_workers,
            loss_weighting=args.dsbm_loss_weighting,
        )
        trainer, completed_epochs = train_pairwise_dsbm(
            train_source,
            train_target,
            hyperparams=hyperparams,
            epochs=args.epochs,
            train_seconds=args.train_seconds,
            device=device,
            progress_label=f"{ALGORITHM_LABELS[algorithm_key]} seed={seed}",
            quality_eval_every=args.quality_eval_every,
            checkpoint_callback=dsbm_checkpoint_callback if args.quality_eval_every > 0 else None,
            stop_callback=lambda _updates: checkpoint_recorder.elapsed_seconds() >= args.train_seconds,
        )
        elapsed_seconds = checkpoint_recorder.elapsed_seconds()
        if args.final_eval:
            predicted = sample_dsbm_target(
                trainer,
                eval_source,
                rollout_batch_size=args.rollout_batch_size,
            )
        final_prediction_source = snapshot_source if snapshot_source is not None else eval_source
        final_snapshot = checkpoint_recorder.time_block(
            lambda: save_final_outputs(
                sample_dsbm_target(
                    trainer,
                    final_prediction_source,
                    rollout_batch_size=args.rollout_batch_size,
                ),
                epoch=completed_epochs,
                elapsed_seconds=elapsed_seconds,
            )
        )
    elif algorithm_key == "lightsb_m":

        def predict_output(model, source_points: torch.Tensor) -> torch.Tensor:
            return sample_lightsb_target(
                model,
                source_points,
                sigma=args.sigma,
                rollout_batch_size=args.rollout_batch_size,
                device=device,
            )

        def callback(epoch: int, model) -> bool:
            if args.quality_eval_every > 0:
                checkpoint_mmd(predict_output(model, eval_source), epoch=epoch)
            return checkpoint_recorder.elapsed_seconds() >= args.train_seconds

        model, completed_epochs = train_pairwise_lightsb_m(
            train_source,
            train_target,
            sigma=args.sigma,
            input_plan=args.lightsb_m_input_plan,
            ot_method=args.ot_method,
            batch_size=args.batch_size,
            epochs=args.epochs,
            train_seconds=args.train_seconds,
            num_components=args.lightsb_m_num_components,
            point_chunk_size=args.lightsb_m_point_chunk_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            min_cov_scale=args.lightsb_m_min_cov_scale,
            device=device,
            progress_label=f"{ALGORITHM_LABELS[algorithm_key]} seed={seed}",
            quality_eval_every=args.quality_eval_every,
            checkpoint_callback=None,
            epoch_callback=callback,
        )
        elapsed_seconds = checkpoint_recorder.elapsed_seconds()
        if args.final_eval:
            predicted = predict_output(model, eval_source)
        final_prediction_source = snapshot_source if snapshot_source is not None else eval_source
        final_snapshot = checkpoint_recorder.time_block(
            lambda: save_final_outputs(
                predict_output(model, final_prediction_source),
                epoch=completed_epochs,
                elapsed_seconds=elapsed_seconds,
            )
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm_key}")

    w1 = None
    mmd = None
    if args.final_eval:
        if quality_mmd_gamma is None:
            raise RuntimeError("Final evaluation requested without an MMD gamma estimate.")
        if predicted is None:
            raise RuntimeError("Final evaluation requested but no final prediction was computed.")
        max_eval_points = None if args.max_eval_points <= 0 else args.max_eval_points
        w1 = compute_w1(
            predicted,
            eval_target,
            max_points=max_eval_points,
            method=args.w1_method,
            reg=args.w1_reg,
        )
        mmd = compute_mmd(
            predicted,
            eval_target,
            gamma=quality_mmd_gamma,
            max_points=max_eval_points,
        )
    return {
        "seed": seed,
        "epochs_completed": int(completed_epochs),
        "elapsed_seconds": float(elapsed_seconds),
        "w1": None if w1 is None else float(w1),
        "mmd": None if mmd is None else float(mmd),
        "quality_curve": checkpoint_recorder.checkpoints,
        "saved_predicted_latents": final_snapshot["saved_predicted_latents"],
        "saved_predicted_images": final_snapshot["saved_predicted_images"],
        "saved_source_images": final_snapshot["saved_source_images"],
    }


def benchmark_algorithm(
    algorithm_key: str,
    *,
    dataset,
    alae: ALAEInference | None,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    seed_results = []
    for seed in args.seeds:
        seed_result = run_single_seed(
            algorithm_key,
            train_source=dataset.train_source,
            train_target=dataset.train_target,
            eval_source=dataset.eval_source,
            eval_target=dataset.eval_target,
            eval_source_paths=dataset.eval_source_paths,
            alae=alae,
            args=args,
            device=device,
            seed=seed,
        )
        seed_results.append(seed_result)
        if args.final_eval:
            print(
                f"  seed {seed}: "
                f"W1={seed_result['w1']:.6f} | "
                f"MMD={seed_result['mmd']:.6f} | "
                f"epochs={seed_result['epochs_completed']}"
            )
        else:
            print(f"  seed {seed}: elapsed={seed_result['elapsed_seconds']:.2f}s | epochs={seed_result['epochs_completed']}")

    elapsed_values = np.asarray([entry["elapsed_seconds"]
                                for entry in seed_results], dtype=np.float64)
    epoch_values = np.asarray([entry["epochs_completed"] for entry in seed_results], dtype=np.float64)
    if args.final_eval:
        w1_values = np.asarray([entry["w1"]
                               for entry in seed_results], dtype=np.float64)
        mmd_values = np.asarray([entry["mmd"]
                                for entry in seed_results], dtype=np.float64)
        mean_w1 = float(w1_values.mean())
        std_w1 = float(w1_values.std(ddof=0))
        mean_mmd = float(mmd_values.mean())
        std_mmd = float(mmd_values.std(ddof=0))
    else:
        mean_w1 = None
        std_w1 = None
        mean_mmd = None
        std_mmd = None
    return {
        "algorithm": ALGORITHM_LABELS[algorithm_key],
        "algorithm_key": algorithm_key,
        "dataset": f"{dataset.source_label}->{dataset.target_label}",
        "source_label": dataset.source_label,
        "target_label": dataset.target_label,
        "latent_dim": dataset.latent_dim,
        "artifact_desc": dataset.artifact_desc,
        "mean_w1": mean_w1,
        "std_w1": std_w1,
        "mean_mmd": mean_mmd,
        "std_mmd": std_mmd,
        "mean_elapsed_seconds": float(elapsed_values.mean()),
        "std_elapsed_seconds": float(elapsed_values.std(ddof=0)),
        "mean_epochs_completed": float(epoch_values.mean()),
        "std_epochs_completed": float(epoch_values.std(ddof=0)),
        "quality_curve_metric": QUALITY_CURVE_METRIC,
        "quality_curve": summarize_quality_curve(seed_results),
        "seed_results": seed_results,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = resolve_device(args.device)
    dataset, alae = load_dataset(args, device=device)

    print(f"Using device: {device}")
    print(
        f"Algorithms: {', '.join(ALGORITHM_LABELS[key] for key in args.algorithms)}")
    print(
        f"Translation task: {dataset.source_label} -> {dataset.target_label}")
    print(f"Latent artifact: {dataset.artifact_desc}")
    print(f"Latent dimension: {dataset.latent_dim}")
    if alae is not None:
        if use_default_ffhq_image_inputs(args):
            print(f"FFHQ root: {args.ffhq_root}")
            print(
                "FFHQ age groups: "
                f"adult={tuple(args.ffhq_adult_age_groups)}, child={tuple(args.ffhq_child_age_groups)}"
            )
        print(f"ALAE config: {alae.config.config_path}")
        print(f"ALAE checkpoint: {alae.config.checkpoint_path}")
        print(f"ALAE resolution: {alae.resolution}")
    print(
        f"Train source/target: {dataset.train_source.shape[0]} / {dataset.train_target.shape[0]}")
    print(
        f"Eval source/target: {dataset.eval_source.shape[0]} / {dataset.eval_target.shape[0]}")
    print(f"Sigma: {args.sigma}")
    print(f"OT method: {args.ot_method}")
    if args.final_eval and args.max_eval_points <= 0:
        print("Evaluating final latent-space W1/MMD on the full evaluation populations.")
    elif args.final_eval:
        print(
            f"Evaluating final latent-space W1/MMD with a {args.max_eval_points}-point cap.")
    else:
        print("Final latent-space W1/MMD evaluation disabled.")
    if args.quality_eval_every > 0:
        print(
            f"Recording untimed latent-space MMD checkpoints every {args.quality_eval_every} epochs.")
    else:
        print("Periodic latent-space MMD checkpoints disabled.")
    print(f"Timed training budget per algorithm/seed: {args.train_seconds:g}s.")
    if args.epochs > 0:
        print(f"Epoch cap: {args.epochs}")
    else:
        print("Epoch cap disabled.")
    if args.save_sample_images_dir is not None:
        print(
            f"Saving {args.num_saved_images} decoded images per algorithm/seed to {args.save_sample_images_dir}.")
    elif args.save_sample_latents_dir is not None:
        print(f"Saving {args.num_saved_samples} translated latent samples per algorithm/seed.")

    results = []
    for algorithm_key in args.algorithms:
        print(f"\nAlgorithm: {ALGORITHM_LABELS[algorithm_key]}")
        result = benchmark_algorithm(
            algorithm_key,
            dataset=dataset,
            alae=alae,
            args=args,
            device=device,
        )
        results.append(result)

    print("\nImage latent translation summary")
    for result in results:
        if args.final_eval:
            print(
                f"  {result['algorithm']} | "
                f"W1={format_metric(result['mean_w1'], result['std_w1'])} | "
                f"MMD={format_metric(result['mean_mmd'], result['std_mmd'])} | "
                f"Time={result['mean_elapsed_seconds']:.2f}s | "
                f"Epochs={result['mean_epochs_completed']:.1f}"
            )
        else:
            print(
                f"  {result['algorithm']} | Time={result['mean_elapsed_seconds']:.2f}s | "
                f"Epochs={result['mean_epochs_completed']:.1f}"
            )

    if args.output_json is not None:
        payload = {
            "config": {
                "algorithms": args.algorithms,
                "seeds": args.seeds,
                "source_label": args.source_label,
                "target_label": args.target_label,
                "sigma": args.sigma,
                "ot_method": args.ot_method,
                "batch_size": args.batch_size,
                "train_seconds": args.train_seconds,
                "epochs": args.epochs,
                "quality_eval_every": args.quality_eval_every,
                "width": args.width,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "steps_per_unit": args.steps_per_unit,
                "rollout_batch_size": args.rollout_batch_size,
                "final_eval": args.final_eval,
                "max_eval_points": args.max_eval_points,
                "w1_method": args.w1_method,
                "w1_reg": args.w1_reg,
                "device": str(device),
                "input_mode": "latents" if using_latent_inputs(args) else "raw_images",
                "ffhq_root": str(args.ffhq_root),
                "ffhq_child_age_groups": args.ffhq_child_age_groups,
                "ffhq_adult_age_groups": args.ffhq_adult_age_groups,
                "ffhq_eval_fraction": args.ffhq_eval_fraction,
                "ffhq_split_seed": args.ffhq_split_seed,
                "source_train_dir": None if args.source_train_dir is None else str(args.source_train_dir),
                "target_train_dir": None if args.target_train_dir is None else str(args.target_train_dir),
                "source_eval_dir": None if args.source_eval_dir is None else str(args.source_eval_dir),
                "target_eval_dir": None if args.target_eval_dir is None else str(args.target_eval_dir),
                "alae_root": str(args.alae_root),
                "alae_config": args.alae_config,
                "alae_checkpoint": None if args.alae_checkpoint is None else str(args.alae_checkpoint),
                "alae_encode_batch_size": args.alae_encode_batch_size,
                "save_sample_latents_dir": None if args.save_sample_latents_dir is None else str(args.save_sample_latents_dir),
                "num_saved_samples": args.num_saved_samples,
                "save_sample_images_dir": None if args.save_sample_images_dir is None else str(args.save_sample_images_dir),
                "num_saved_images": args.num_saved_images,
                "coverage_anchors": args.coverage_anchors,
                "coverage_anchor_selection": args.coverage_anchor_selection,
                "coverage_anchor_gon_plus_candidates": args.coverage_anchor_gon_plus_candidates,
                "coverage_anchor_weight_temperature": args.coverage_anchor_weight_temperature,
                "coverage_anchor_refresh_epochs": args.coverage_anchor_refresh_epochs,
                "coverage_point_match_mode": args.coverage_point_match_mode,
                "dsb_num_iter": args.dsb_num_iter,
                "dsb_n_ipf": args.dsb_n_ipf,
                "dsb_num_steps": args.dsb_num_steps,
                "dsb_gamma_min": args.dsb_gamma_min,
                "dsb_gamma_max": args.dsb_gamma_max,
                "dsb_gamma_space": args.dsb_gamma_space,
                "dsb_cache_batch_size": args.dsb_cache_batch_size,
                "dsb_num_cache_batches": args.dsb_num_cache_batches,
                "dsb_time_embed_dim": args.dsb_time_embed_dim,
                "dsb_num_workers": args.dsb_num_workers,
                "dsbm_num_iter": args.dsbm_num_iter,
                "dsbm_n_outer": args.dsbm_n_outer,
                "dsbm_cache_batch_size": args.dsbm_cache_batch_size,
                "dsbm_num_cache_batches": args.dsbm_num_cache_batches,
                "dsbm_time_embed_dim": args.dsbm_time_embed_dim,
                "dsbm_num_workers": args.dsbm_num_workers,
                "dsbm_loss_weighting": args.dsbm_loss_weighting,
                "lightsb_m_num_components": args.lightsb_m_num_components,
                "lightsb_m_point_chunk_size": args.lightsb_m_point_chunk_size,
                "lightsb_m_input_plan": args.lightsb_m_input_plan,
                "lightsb_m_min_cov_scale": args.lightsb_m_min_cov_scale,
                "artifact_desc": dataset.artifact_desc,
                "latent_dim": dataset.latent_dim,
            },
            "results": results,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved raw metrics to {args.output_json}")


if __name__ == "__main__":
    main()
