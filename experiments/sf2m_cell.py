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
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from cell_exp.common import (build_leaveout_pair_indices, build_model_times,
                             format_metric, resolve_dataset_sigma,
                             resolve_device, sample_minibatch, set_seed)
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


def train_leave_one_out_sf2m(
    timepoints: list[torch.Tensor],
    model_times: np.ndarray,
    *,
    missing_index: int,
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

                x0 = sample_minibatch(
                    timepoints[src_idx], batch_size).to(device)
                x1 = sample_minibatch(
                    timepoints[dst_idx], batch_size).to(device)
                tau, xt, ut, eps = flow_matcher.sample_location_and_conditional_flow(
                    x0,
                    x1,
                    return_noise=True,
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
