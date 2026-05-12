"""Reproduce the SF2M-Exact row from Table 4 in ``res/sf2m.pdf``.

This script targets the low-dimensional single-cell interpolation benchmark:
- datasets: EB, Cite, Multi
- representation: 5 PCs
- metric: leave-one-timepoint-out 1-Wasserstein distance
- method: exact-OT SF2M with shared parameters across observed intervals

By default it reads the real datasets shipped in this repo:
- ``data/embryoid/scRNAseq`` for EB
- ``data/cite_multi`` for Cite and Multi

Example:
    python experiments/sf2m_cell.py --datasets eb cite multi --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from cell_exp.common import (build_leaveout_pair_indices, build_model_times,
                             format_metric, resolve_dataset_sigma,
                             resolve_device, set_seed)
from cell_exp.data import (DEFAULT_DONOR, DEFAULT_PCA_EMBED_DIM,
                           TABLE4_DATASETS, TABLE4_ORDER, Table4DatasetSpec,
                           load_real_dataset)
from cell_exp.quality import (DEFAULT_QUALITY_CURVE_EVAL_POINTS,
                              DEFAULT_QUALITY_EVAL_EVERY, QUALITY_CURVE_METRIC,
                              QualityCheckpointRecorder, estimate_mmd_gamma,
                              summarize_quality_curve)
from cell_exp.sf2m import (build_models, evaluate_leave_one_out,
                           evaluate_leave_one_out_mmd, make_time_input)
from torchcfm.conditional_flow_matching import \
    SchrodingerBridgeConditionalFlowMatcher
from torchcfm.models import MLP
from tqdm import tqdm

EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))


@dataclass(frozen=True)
class CoverageAccelerationPlan:
    anchor_plan: torch.Tensor
    source_anchor_to_point_ind: dict[int, list[int]]
    target_anchor_to_point_ind: dict[int, list[int]]
    matched_point_plans: dict[tuple[int, int], "PointMatchPlan"] | None = None


@dataclass(frozen=True)
class PointMatchPlan:
    pair_probs: torch.Tensor
    source_point_indices: torch.Tensor
    target_point_indices: torch.Tensor


DEFAULT_COVERAGE_ANCHORS = 256
DEFAULT_COVERAGE_ANCHOR_REFRESH_EPOCHS = 100


def farthest_first_k_center(
    dataset: torch.Tensor,
    k: int,
    *,
    initial_center: int | None = None,
) -> torch.Tensor:
    if dataset.ndim != 2:
        raise ValueError(
            "Expected dataset to have shape [n_samples, n_features].")
    if k <= 0:
        raise ValueError("k must be positive.")

    effective_k = min(k, dataset.shape[0])
    if initial_center is None:
        initial_center = random.randrange(dataset.shape[0])
    centers = [initial_center]
    min_distances = torch.cdist(dataset[[initial_center]], dataset).squeeze(0)

    for _ in range(1, effective_k):
        next_center = torch.argmax(min_distances).item()
        centers.append(next_center)
        distances_to_new_center = torch.cdist(
            dataset[[next_center]], dataset).squeeze(0)
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
    if num_candidates <= 0:
        raise ValueError("num_candidates must be positive.")

    effective_candidates = min(num_candidates, dataset.shape[0])
    candidate_indices = random.sample(
        range(dataset.shape[0]),
        k=effective_candidates,
    )
    best_anchors: torch.Tensor | None = None
    best_radius = float("inf")
    for candidate_index in candidate_indices:
        anchors = farthest_first_k_center(
            dataset,
            k=k,
            initial_center=candidate_index,
        )
        radius = anchor_cover_radius(dataset, anchors)
        if radius < best_radius:
            best_radius = radius
            best_anchors = anchors
    if best_anchors is None:
        raise RuntimeError("Gon+ failed to produce any anchor candidates.")
    return best_anchors


def anchors_and_weights(
    dataset: torch.Tensor,
    k: int,
    *,
    method: str = "gon",
    gon_plus_candidates: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if method == "gon":
        anchors = farthest_first_k_center(dataset, k=k)
    elif method == "gon_plus":
        anchors = gon_plus_k_center(
            dataset,
            k=k,
            num_candidates=gon_plus_candidates,
        )
    else:
        raise ValueError(f"Unsupported anchor selection method: {method}")
    distances = torch.cdist(dataset, anchors)
    assignments = torch.argmin(distances, dim=1)
    counts = torch.bincount(assignments, minlength=anchors.shape[0])
    weights = counts.float() / dataset.shape[0]
    return anchors, weights, assignments


def temperature_scale_probs(probs: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    probs = probs.to(dtype=torch.float32).clamp_min(1e-12)
    if temperature == 1.0:
        return probs / probs.sum().clamp_min(1e-12)
    scaled = probs.pow(1.0 / temperature)
    return scaled / scaled.sum().clamp_min(1e-12)


def sample_anchors(
    anchor_probs: torch.Tensor,
    batch_size: int,
    *,
    replace: bool = True,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    p = anchor_probs.flatten().clamp_min(0)
    total_mass = p.sum()
    if total_mass <= 0:
        raise RuntimeError("Anchor transport plan has zero total mass.")
    p = temperature_scale_probs(p / total_mass, temperature)
    choices = torch.multinomial(
        input=p,
        num_samples=batch_size,
        replacement=replace,
    )
    return (
        torch.div(choices, anchor_probs.shape[1], rounding_mode="floor"),
        torch.remainder(choices, anchor_probs.shape[1]),
    )


def anchor_cell_map(assignments: torch.Tensor) -> dict[int, list[int]]:
    anchor_to_point_ind: dict[int, list[int]] = {}
    for point_index, anchor_index in enumerate(assignments.tolist()):
        if anchor_index in anchor_to_point_ind:
            anchor_to_point_ind[anchor_index].append(point_index)
        else:
            anchor_to_point_ind[anchor_index] = [point_index]
    return anchor_to_point_ind


def anchor_idx_to_rand_point_in_cell(
    anchor_ind: torch.Tensor,
    anchor_to_point_ind: dict[int, list[int]],
    dataset: torch.Tensor,
) -> torch.Tensor:
    point_indices = [
        random.choice(anchor_to_point_ind[int(anchor_idx)])
        for anchor_idx in anchor_ind.tolist()
    ]
    return dataset[torch.tensor(point_indices, dtype=torch.long, device=dataset.device)]


def solve_ot_plan(
    source_weights: torch.Tensor,
    target_weights: torch.Tensor,
    cost: torch.Tensor,
    *,
    flow_matcher: SchrodingerBridgeConditionalFlowMatcher,
    error_context: str,
) -> torch.Tensor:
    if flow_matcher.ot_sampler.normalize_cost:
        cost = cost / cost.max().clamp_min(1e-12)
    plan = flow_matcher.ot_sampler.ot_fn(
        source_weights.cpu().numpy(),
        target_weights.cpu().numpy(),
        cost.cpu().numpy(),
    )
    plan = torch.from_numpy(plan).to(dtype=torch.float32)
    if not torch.all(torch.isfinite(plan)):
        raise RuntimeError(f"{error_context} produced a non-finite OT plan.")
    plan = plan.clamp_min_(0)
    if torch.abs(plan.sum()) < 1e-8:
        raise RuntimeError(f"{error_context} produced a degenerate OT plan.")
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
    source_weights = torch.full(
        (source_subset.shape[0],),
        1.0 / source_subset.shape[0],
        dtype=torch.float32,
    )
    target_weights = torch.full(
        (target_subset.shape[0],),
        1.0 / target_subset.shape[0],
        dtype=torch.float32,
    )
    point_plan = solve_ot_plan(
        source_weights,
        target_weights,
        torch.cdist(source_subset, target_subset) ** 2,
        flow_matcher=flow_matcher,
        error_context="Coverage acceleration point-matching",
    )
    support = torch.nonzero(point_plan > 0, as_tuple=False)
    if support.numel() == 0:
        raise RuntimeError(
            "Coverage acceleration point-matching produced an empty support.")
    pair_probs = point_plan[support[:, 0], support[:, 1]]
    pair_probs = pair_probs / pair_probs.sum().clamp_min(1e-12)
    source_indices = torch.tensor(
        source_point_indices, dtype=torch.long)[support[:, 0]]
    target_indices = torch.tensor(
        target_point_indices, dtype=torch.long)[support[:, 1]]
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
        point_plan = coverage_plan.matched_point_plans.get(
            (source_anchor, target_anchor)
        )
        if point_plan is None:
            raise RuntimeError(
                "Missing cached point match plan for anchor pair "
                f"({source_anchor}, {target_anchor})."
            )
        choices = torch.multinomial(
            point_plan.pair_probs,
            num_samples=num_samples,
            replacement=True,
        )
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
        error_context="Coverage acceleration anchor-matching",
    )
    source_anchor_to_point_ind = anchor_cell_map(source_assignments)
    target_anchor_to_point_ind = anchor_cell_map(target_assignments)
    matched_point_plans: dict[tuple[int, int], PointMatchPlan] | None = None
    if point_match_mode == "ot":
        matched_point_plans = {}
        for source_anchor, target_anchor in torch.nonzero(
            anchor_plan > 0, as_tuple=False
        ).tolist():
            matched_point_plans[(source_anchor, target_anchor)] = (
                build_point_match_plan(
                    source_points,
                    target_points,
                    source_point_indices=source_anchor_to_point_ind[source_anchor],
                    target_point_indices=target_anchor_to_point_ind[target_anchor],
                    flow_matcher=flow_matcher,
                )
            )

    return CoverageAccelerationPlan(
        anchor_plan=anchor_plan,
        source_anchor_to_point_ind=source_anchor_to_point_ind,
        target_anchor_to_point_ind=target_anchor_to_point_ind,
        matched_point_plans=matched_point_plans,
    )


def train_leave_one_out_sf2m(
    timepoints: list[torch.Tensor],
    model_times: np.ndarray,
    *,
    missing_index: int,
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
) -> tuple[MLP, MLP]:
    dim = int(timepoints[0].shape[1])
    flow_model, score_model = build_models(dim=dim, width=width, device=device)
    optimizer = torch.optim.AdamW(
        list(flow_model.parameters()) + list(score_model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    flow_matcher = SchrodingerBridgeConditionalFlowMatcher(
        sigma=sigma, ot_method=ot_method)
    pair_indices = build_leaveout_pair_indices(
        num_timepoints=len(timepoints),
        missing_index=missing_index,
    )

    def refresh_coverage_plans() -> dict[tuple[int, int], CoverageAccelerationPlan]:
        return {
            (src_idx, dst_idx): build_coverage_acceleration_plan(
                timepoints[src_idx],
                timepoints[dst_idx],
                num_anchors=coverage_anchors,
                anchor_selection=coverage_anchor_selection,
                gon_plus_candidates=coverage_anchor_gon_plus_candidates,
                flow_matcher=flow_matcher,
                point_match_mode=coverage_point_match_mode,
            )
            for src_idx, dst_idx in pair_indices
        }

    coverage_plans = refresh_coverage_plans()
    max_points = max(
        max(
            timepoints[src_idx].shape[0],
            timepoints[dst_idx].shape[0],
        )
        for src_idx, dst_idx in pair_indices
    )

    steps_per_epoch = max(1, math.ceil(max_points / batch_size))
    progress = tqdm(range(epochs), desc=progress_label,
                    leave=False, dynamic_ncols=True)
    running_loss = None

    for epoch_idx in progress:
        if coverage_anchor_refresh_epochs > 0 and epoch_idx > 0 and epoch_idx % coverage_anchor_refresh_epochs == 0:
            coverage_plans = refresh_coverage_plans()

        epoch_loss = 0.0
        n_updates = 0
        flow_model.train()
        score_model.train()

        for _ in range(steps_per_epoch):
            local_ts = []
            global_ts = []
            xts = []
            uts = []
            epss = []

            for src_idx, dst_idx in pair_indices:
                start_time = float(model_times[src_idx])
                end_time = float(model_times[dst_idx])
                interval_scale = end_time - start_time
                if interval_scale <= 0:
                    raise ValueError(
                        "Model times must be strictly increasing.")

                coverage_plan = coverage_plans[(src_idx, dst_idx)]
                sampled_source_anchors, sampled_target_anchors = sample_anchors(
                    coverage_plan.anchor_plan,
                    batch_size=batch_size,
                    replace=True,
                    temperature=coverage_anchor_weight_temperature,
                )
                if coverage_point_match_mode == "ot":
                    x0, x1 = sample_matched_points_from_ot_plans(
                        sampled_source_anchors,
                        sampled_target_anchors,
                        coverage_plan=coverage_plan,
                        source_dataset=timepoints[src_idx],
                        target_dataset=timepoints[dst_idx],
                    )
                elif coverage_point_match_mode == "random":
                    x0 = anchor_idx_to_rand_point_in_cell(
                        sampled_source_anchors,
                        coverage_plan.source_anchor_to_point_ind,
                        timepoints[src_idx],
                    )
                    x1 = anchor_idx_to_rand_point_in_cell(
                        sampled_target_anchors,
                        coverage_plan.target_anchor_to_point_ind,
                        timepoints[dst_idx],
                    )
                else:
                    raise ValueError(
                        "Unsupported coverage point match mode: "
                        f"{coverage_point_match_mode}"
                    )
                x0 = x0.to(device)
                x1 = x1.to(device)
                tau, xt, ut, eps = flow_matcher.sample_location_and_conditional_flow(
                    x0,
                    x1,
                    return_noise=True,
                    use_ot=False,
                )

                local_ts.append(tau)
                global_ts.append(tau * interval_scale + start_time)
                xts.append(xt)
                uts.append(ut / interval_scale)
                epss.append(eps)

            local_t = torch.cat(local_ts, dim=0)
            global_t = torch.cat(global_ts, dim=0)
            xt = torch.cat(xts, dim=0)
            ut = torch.cat(uts, dim=0)
            eps = torch.cat(epss, dim=0)

            xt_with_t = make_time_input(xt, global_t)
            vt = flow_model(xt_with_t)
            st = score_model(xt_with_t)
            lambda_t = flow_matcher.compute_lambda(local_t)

            flow_loss = torch.mean((vt - ut) ** 2)
            score_loss = torch.mean((lambda_t[:, None] * st + eps) ** 2)
            loss = flow_loss + score_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_updates += 1

        mean_loss = epoch_loss / max(n_updates, 1)
        if running_loss is None:
            running_loss = mean_loss
        else:
            running_loss = 0.95 * running_loss + 0.05 * mean_loss
        progress.set_postfix(loss=f"{running_loss:.5f}")

        epoch_number = epoch_idx + 1
        should_record = (
            checkpoint_callback is not None
            and quality_eval_every > 0
            and (
                epoch_number % quality_eval_every == 0
                or epoch_number == epochs
            )
        )
        if should_record:
            checkpoint_callback(epoch_number, flow_model, score_model)

    return flow_model, score_model


def benchmark_dataset(
    spec: Table4DatasetSpec,
    *,
    data_root: Path,
    dims: int,
    pca_embed_dim: int,
    fit_pca: bool,
    whiten: bool,
    pca_batch_size: int,
    donor: int,
    coverage_anchors: int,
    coverage_anchor_selection: str,
    coverage_anchor_gon_plus_candidates: int,
    coverage_anchor_weight_temperature: float,
    coverage_anchor_refresh_epochs: int,
    coverage_point_match_mode: str,
    seeds: list[int],
    sigma: float,
    ot_method: str,
    batch_size: int,
    epochs: int,
    width: int,
    lr: float,
    weight_decay: float,
    steps_per_unit: int,
    rollout_batch_size: int,
    device: torch.device,
    max_eval_points: int | None,
    quality_eval_every: int,
    w1_method: str,
    w1_reg: float,
    time_mode: str,
) -> dict[str, object]:
    timepoints, raw_times, artifact_desc = load_real_dataset(
        spec,
        data_root=data_root,
        dims=dims,
        pca_embed_dim=pca_embed_dim,
        fit_pca=fit_pca,
        whiten=whiten,
        pca_batch_size=pca_batch_size,
        donor=donor,
    )
    model_times = build_model_times(raw_times, time_mode)
    seed_results = []

    for seed in seeds:
        set_seed(seed)
        leave_out_metrics = []
        for missing_index in spec.leave_out:
            checkpoint_recorder = QualityCheckpointRecorder()
            quality_mmd_gamma = estimate_mmd_gamma(timepoints[missing_index])

            def checkpoint_callback(
                epoch_number: int,
                current_flow_model: MLP,
                current_score_model: MLP,
            ) -> None:
                checkpoint_recorder.time_evaluation(
                    epoch=epoch_number,
                    evaluate=lambda: evaluate_leave_one_out_mmd(
                        current_flow_model,
                        current_score_model,
                        timepoints,
                        model_times,
                        missing_index=missing_index,
                        sigma=sigma,
                        steps_per_unit=steps_per_unit,
                        rollout_batch_size=rollout_batch_size,
                        device=device,
                        gamma=quality_mmd_gamma,
                        max_eval_points=DEFAULT_QUALITY_CURVE_EVAL_POINTS,
                    ),
                )

            flow_model, score_model = train_leave_one_out_sf2m(
                timepoints,
                model_times,
                missing_index=missing_index,
                coverage_anchors=coverage_anchors,
                coverage_anchor_selection=coverage_anchor_selection,
                coverage_anchor_gon_plus_candidates=coverage_anchor_gon_plus_candidates,
                coverage_anchor_weight_temperature=coverage_anchor_weight_temperature,
                coverage_anchor_refresh_epochs=coverage_anchor_refresh_epochs,
                coverage_point_match_mode=coverage_point_match_mode,
                sigma=sigma,
                ot_method=ot_method,
                batch_size=batch_size,
                epochs=epochs,
                width=width,
                lr=lr,
                weight_decay=weight_decay,
                device=device,
                progress_label=f"{spec.label} seed={seed} miss={missing_index}",
                quality_eval_every=quality_eval_every,
                checkpoint_callback=checkpoint_callback,
            )
            w1 = evaluate_leave_one_out(
                flow_model,
                score_model,
                timepoints,
                model_times,
                missing_index=missing_index,
                sigma=sigma,
                steps_per_unit=steps_per_unit,
                rollout_batch_size=rollout_batch_size,
                device=device,
                max_eval_points=max_eval_points,
                w1_method=w1_method,
                w1_reg=w1_reg,
            )
            mmd = evaluate_leave_one_out_mmd(
                flow_model,
                score_model,
                timepoints,
                model_times,
                missing_index=missing_index,
                sigma=sigma,
                steps_per_unit=steps_per_unit,
                rollout_batch_size=rollout_batch_size,
                device=device,
                gamma=quality_mmd_gamma,
                max_eval_points=max_eval_points,
            )
            leave_out_metrics.append(
                {
                    "missing_index": missing_index,
                    "w1": float(w1),
                    "mmd": float(mmd),
                    "quality_curve": checkpoint_recorder.checkpoints,
                }
            )

        mean_w1 = float(np.mean([entry["w1"] for entry in leave_out_metrics]))
        mean_mmd = float(np.mean([entry["mmd"]
                         for entry in leave_out_metrics]))
        seed_results.append(
            {
                "seed": seed,
                "mean_w1": mean_w1,
                "mean_mmd": mean_mmd,
                "leave_out": leave_out_metrics,
            }
        )
        print(f"  seed {seed}: W1={mean_w1:.6f} | MMD={mean_mmd:.6f}")

    w1_means = np.asarray([entry["mean_w1"]
                          for entry in seed_results], dtype=np.float64)
    mmd_means = np.asarray(
        [entry["mean_mmd"] for entry in seed_results],
        dtype=np.float64,
    )
    return {
        "dataset": spec.label,
        "artifact": artifact_desc,
        "times": raw_times.tolist(),
        "model_times": model_times.tolist(),
        "mean_w1": float(w1_means.mean()),
        "std_w1": float(w1_means.std(ddof=0)),
        "mean_mmd": float(mmd_means.mean()),
        "std_mmd": float(mmd_means.std(ddof=0)),
        "quality_curve_metric": QUALITY_CURVE_METRIC,
        "quality_curve": summarize_quality_curve(seed_results),
        "seed_results": seed_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the SF2M-Exact row from Table 4 in res/sf2m.pdf.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["eb", "cite", "multi"],
        choices=list(TABLE4_ORDER),
        help="Which Table 4 datasets to run.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Repository data root containing cite_multi and embryoid.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Model seeds used for the paper summary.",
    )
    parser.add_argument("--dims", type=int, default=5,
                        help="Number of feature dimensions to use.")
    parser.add_argument(
        "--pca-embed-dim",
        type=int,
        default=DEFAULT_PCA_EMBED_DIM,
        help=(
            "Number of PCA components to build before whitening/truncating. "
            "Used only when fitting PCA from the raw local data."
        ),
    )
    parser.set_defaults(fit_pca=True)
    parser.add_argument(
        "--no-pca",
        dest="fit_pca",
        action="store_false",
        help="Skip PCA and use the first --dims features directly.",
    )
    parser.add_argument(
        "--no-whiten",
        action="store_true",
        help="Disable per-dimension whitening after truncation/PCA.",
    )
    parser.add_argument(
        "--pca-batch-size",
        type=int,
        default=512,
        help="Batch size used when fitting/transfoming PCA from raw data.",
    )
    parser.add_argument(
        "--donor",
        type=int,
        default=DEFAULT_DONOR,
        help="Donor id used for Cite and Multi.",
    )
    parser.add_argument(
        "--coverage-anchors",
        type=int,
        default=DEFAULT_COVERAGE_ANCHORS,
        help="Number of coverage-acceleration anchors per marginal timepoint.",
    )
    parser.add_argument(
        "--coverage-anchor-selection",
        type=str,
        default="gon",
        choices=["gon", "gon_plus"],
        help=(
            "How anchors are selected. 'gon' uses the current Gonzalez "
            "farthest-first traversal; 'gon_plus' evaluates multiple randomized "
            "Gonzalez starts and keeps the best covering radius."
        ),
    )
    parser.add_argument(
        "--coverage-anchor-gon-plus-candidates",
        type=int,
        default=3,
        help="Number of randomized Gonzalez starts to try for --coverage-anchor-selection=gon_plus.",
    )
    parser.add_argument(
        "--coverage-anchor-weight-temperature",
        type=float,
        default=1.0,
        help=(
            "Temperature used to flatten anchor-pair sampling after the "
            "anchor-level OT solve. T=1 leaves sampling from the OT plan unchanged."
        ),
    )
    parser.add_argument(
        "--coverage-anchor-refresh-epochs",
        type=int,
        default=DEFAULT_COVERAGE_ANCHOR_REFRESH_EPOCHS,
        help=(
            "Recompute randomized coverage anchors every N epochs. "
            "Use 0 to keep the initial anchors fixed."
        ),
    )
    parser.add_argument(
        "--coverage-point-match-mode",
        type=str,
        default="random",
        choices=["ot", "random"],
        help=(
            "How to sample point pairs after drawing an anchor pair. "
            "'ot' precomputes point-level OT inside each matched anchor pair; "
            "'random' samples points independently within the anchor cells."
        ),
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help=(
            "Global sigma override used for all datasets. "
            "If unset, dataset-specific defaults are used."
        ),
    )
    parser.add_argument(
        "--sigma-eb",
        type=float,
        default=0.25,
        help="Default sigma used for the EB dataset when --sigma is unset.",
    )
    parser.add_argument(
        "--sigma-cite",
        type=float,
        default=0.25,
        help="Default sigma used for the Cite dataset when --sigma is unset.",
    )
    parser.add_argument(
        "--sigma-multi",
        type=float,
        default=1.0,
        help="Default sigma used for the Multi dataset when --sigma is unset.",
    )
    parser.add_argument(
        "--time-mode",
        type=str,
        default="discrete",
        choices=["discrete", "raw", "scaled"],
        help=(
            "Time coordinates used by the shared trajectory model. "
            "'discrete' matches the paper's per-timepoint integration setup."
        ),
    )
    parser.add_argument(
        "--ot-method",
        type=str,
        default="exact",
        choices=["exact", "sinkhorn"],
        help="Static OT method used during training.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument(
        "--quality-eval-every",
        type=int,
        default=DEFAULT_QUALITY_EVAL_EVERY,
        help=(
            "Record elapsed time and MMD every N epochs. "
            "Use 0 to disable periodic quality checkpoints."
        ),
    )
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--steps-per-unit",
        type=int,
        default=100,
        help="Euler-Maruyama steps per unit of normalized time.",
    )
    parser.add_argument("--rollout-batch-size", type=int, default=2048)
    parser.add_argument(
        "--max-eval-points",
        type=int,
        default=0,
        help=(
            "Subsample cap for the 1-Wasserstein evaluation. "
            "Use 0 to evaluate on the full pushed-forward populations."
        ),
    )
    parser.add_argument(
        "--w1-method",
        type=str,
        default="exact",
        choices=["exact", "sinkhorn"],
        help="Method used for the leave-one-out 1-Wasserstein metric.",
    )
    parser.add_argument(
        "--w1-reg",
        type=float,
        default=0.05,
        help="Regularization used when --w1-method=sinkhorn.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    whiten = not args.no_whiten
    print(f"Using device: {device}")
    print(f"Using time mode: {args.time_mode}")
    if args.sigma is None:
        print(
            "Using dataset-specific sigmas: "
            f"EB={args.sigma_eb}, Cite={args.sigma_cite}, Multi={args.sigma_multi}."
        )
    else:
        print(f"Using global sigma override: {args.sigma}.")
    if args.fit_pca:
        print(
            f"Fitting raw-data PCA embeddings with {max(args.dims, args.pca_embed_dim)} components.")
    print(
        f"Using coverage acceleration with {args.coverage_anchors} anchors per marginal.")
    print(
        "Coverage anchor selection: "
        f"{args.coverage_anchor_selection}."
    )
    if args.coverage_anchor_selection == "gon_plus":
        print(
            "Coverage Gon+ candidates: "
            f"{args.coverage_anchor_gon_plus_candidates}."
        )
    print(
        "Coverage anchor weight temperature: "
        f"{args.coverage_anchor_weight_temperature:.3f}."
    )
    print(
        "Coverage point matching mode: "
        f"{args.coverage_point_match_mode}."
    )
    if args.coverage_anchor_refresh_epochs > 0:
        print(
            "Refreshing coverage anchors "
            f"every {args.coverage_anchor_refresh_epochs} epochs."
        )
    else:
        print("Keeping the initial coverage anchors fixed.")
    if args.max_eval_points <= 0:
        print("Evaluating W1 on full pushed-forward populations.")
    else:
        print(f"Evaluating W1 with a {args.max_eval_points}-point cap.")
    if args.quality_eval_every > 0:
        print(
            "Recording quality checkpoints every "
            f"{args.quality_eval_every} epochs using MMD."
        )
    else:
        print("Periodic quality checkpoints disabled.")

    results = []
    for dataset_key in args.datasets:
        spec = TABLE4_DATASETS[dataset_key]
        dataset_sigma = resolve_dataset_sigma(
            dataset_key,
            sigma=args.sigma,
            sigma_eb=args.sigma_eb,
            sigma_cite=args.sigma_cite,
            sigma_multi=args.sigma_multi,
        )
        print(f"\nDataset: {spec.label} (sigma={dataset_sigma})")
        result = benchmark_dataset(
            spec,
            data_root=args.data_root,
            dims=args.dims,
            pca_embed_dim=args.pca_embed_dim,
            fit_pca=args.fit_pca,
            whiten=whiten,
            pca_batch_size=args.pca_batch_size,
            donor=args.donor,
            coverage_anchors=args.coverage_anchors,
            coverage_anchor_selection=args.coverage_anchor_selection,
            coverage_anchor_gon_plus_candidates=args.coverage_anchor_gon_plus_candidates,
            coverage_anchor_weight_temperature=args.coverage_anchor_weight_temperature,
            coverage_anchor_refresh_epochs=args.coverage_anchor_refresh_epochs,
            coverage_point_match_mode=args.coverage_point_match_mode,
            seeds=args.seeds,
            sigma=dataset_sigma,
            ot_method=args.ot_method,
            batch_size=args.batch_size,
            epochs=args.epochs,
            width=args.width,
            lr=args.lr,
            weight_decay=args.weight_decay,
            steps_per_unit=args.steps_per_unit,
            rollout_batch_size=args.rollout_batch_size,
            device=device,
            max_eval_points=args.max_eval_points,
            quality_eval_every=args.quality_eval_every,
            w1_method=args.w1_method,
            w1_reg=args.w1_reg,
            time_mode=args.time_mode,
        )
        result["sigma"] = dataset_sigma
        results.append(result)

    print("\nSF2M-Exact summary (Table 4 style)")
    for result in results:
        print(
            f"  {result['dataset']} | "
            f"W1={format_metric(result['mean_w1'], result['std_w1'])} | "
            f"MMD={format_metric(result['mean_mmd'], result['std_mmd'])}"
        )

    if args.output_json is not None:
        payload = {
            "config": {
                "datasets": args.datasets,
                "data_root": str(args.data_root),
                "seeds": args.seeds,
                "dims": args.dims,
                "pca_embed_dim": args.pca_embed_dim,
                "fit_pca": args.fit_pca,
                "whiten": whiten,
                "pca_batch_size": args.pca_batch_size,
                "donor": args.donor,
                "coverage_anchors": args.coverage_anchors,
                "coverage_anchor_selection": args.coverage_anchor_selection,
                "coverage_anchor_gon_plus_candidates": args.coverage_anchor_gon_plus_candidates,
                "coverage_anchor_weight_temperature": args.coverage_anchor_weight_temperature,
                "coverage_anchor_refresh_epochs": args.coverage_anchor_refresh_epochs,
                "coverage_point_match_mode": args.coverage_point_match_mode,
                "sigma": args.sigma,
                "sigma_eb": args.sigma_eb,
                "sigma_cite": args.sigma_cite,
                "sigma_multi": args.sigma_multi,
                "time_mode": args.time_mode,
                "ot_method": args.ot_method,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "quality_eval_every": args.quality_eval_every,
                "width": args.width,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "steps_per_unit": args.steps_per_unit,
                "rollout_batch_size": args.rollout_batch_size,
                "max_eval_points": args.max_eval_points,
                "w1_method": args.w1_method,
                "w1_reg": args.w1_reg,
                "device": str(device),
            },
            "results": results,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved raw metrics to {args.output_json}")


if __name__ == "__main__":
    main()
