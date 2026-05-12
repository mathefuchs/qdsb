from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

MPLCONFIGDIR = Path(tempfile.gettempdir()) / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPLCONFIGDIR))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

matplotlib.use("Agg")

DEFAULT_RESULTS_JSON = Path("results/output_img.json")
DEFAULT_IMAGES_DIR = Path("results/output_img_samples")
DEFAULT_OUTPUT_PDF = Path("plots/output_img_grid.pdf")
DEFAULT_THUMB_SIZE = 512
DEFAULT_INDICES = [5, 10, 34, 47, 14, 19]
FS = 9

ALGORITHM_COLUMNS = (
    ("source", "Source"),
    ("sf2m", "SF2M"),
    ("lightsb_m", "LightSB-M"),
    ("qdsb", "QDSB"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a compact PDF image grid for the FFHQ image experiments.",
    )
    parser.add_argument("--results-json", type=Path, default=DEFAULT_RESULTS_JSON)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PDF)
    parser.add_argument("--thumb-size", type=int, default=DEFAULT_THUMB_SIZE)
    parser.add_argument(
        "--layout",
        choices=("standard", "compact"),
        default="standard",
        help="Layout preset. 'standard' keeps the current 6-image layout; 'compact' fits longer grids.",
    )
    parser.add_argument(
        "--indices",
        nargs="+",
        type=int,
        default=list(DEFAULT_INDICES),
        help="Image indices to show, in the requested order.",
    )
    return parser.parse_args()


def load_results(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def metric_map(payload: dict[str, object]) -> dict[str, tuple[float, float]]:
    output: dict[str, tuple[float, float]] = {}
    for result in payload["results"]:
        output[result["algorithm_key"]] = (
            float(result["mean_elapsed_seconds"]),
            float(result["mean_epochs_completed"]),
        )
    return output


def resolve_source_prefix(images_dir: Path) -> str:
    for algorithm_key, _label in ALGORITHM_COLUMNS[1:]:
        matches = sorted(images_dir.glob(f"{algorithm_key}_seed0_final_source_0000.*"))
        if matches:
            return algorithm_key
    raise FileNotFoundError(
        f"Could not find saved source images in {images_dir}."
    )


def image_path(
    images_dir: Path,
    *,
    column_key: str,
    source_prefix: str,
    row_index: int,
) -> Path:
    if column_key == "source":
        matches = sorted(images_dir.glob(f"{source_prefix}_seed0_final_source_{row_index:04d}.*"))
        if not matches:
            raise FileNotFoundError(f"Missing source image {row_index:04d} in {images_dir}.")
        return matches[0]
    path = images_dir / f"{column_key}_seed0_final_{row_index:04d}.png"
    if not path.exists():
        raise FileNotFoundError(f"Missing generated image {path}.")
    return path


def load_thumbnail(path: Path, *, thumb_size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if image.size != (thumb_size, thumb_size):
        image = image.resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
    return np.asarray(image)


def configure_matplotlib() -> None:
    plt.style.use("tableau-colorblind10")
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Times"],
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts": False,
            "text.latex.preamble": r"\usepackage{times}\usepackage{helvet}\usepackage{courier}",
            "savefig.bbox": "tight",
        }
    )
    plt.rc("xtick", labelsize=FS)
    plt.rc("ytick", labelsize=FS)


def build_footer_text(
    column_key: str,
    stats: dict[str, tuple[float, float]],
    *,
    layout: str,
) -> str:
    if column_key == "source":
        if layout == "compact":
            return "input\n(no train.)"
        return "input\n(no training)"
    elapsed_seconds, epochs = stats[column_key]
    if layout == "compact":
        return f"{elapsed_seconds:.0f}\\,s\n{epochs:.0f} ep."
    return f"{elapsed_seconds:.0f}\\,s\n{epochs:.0f} epochs"


def create_layout(
    *,
    indices: list[int],
    thumb_size: int,
    layout: str,
) -> tuple[plt.Figure, dict[tuple[int, int], plt.Axes], list[float], float, float, float]:
    num_items = len(indices)
    num_panels = 2 if num_items > 3 else 1
    rows_per_panel = int(np.ceil(num_items / num_panels))
    columns_per_panel = len(ALGORITHM_COLUMNS)

    if layout == "compact":
        cell_w = 0.58
        cell_h = 0.58
        row_gap = 0.024
        label_h = 0.13
        footer_h = 0.40
        panel_gap = 0.13 if num_panels == 2 else 0.0
        left_margin = 0.06
        right_margin = 0.03
        top_margin = 0.04
        bottom_margin = 0.06
        footer_offset = 0.014
    else:
        cell_w = 0.62
        cell_h = 0.635
        row_gap = 0.0
        label_h = 0.16
        footer_h = 0.30
        panel_gap = 0.15 if num_panels == 2 else 0.0
        left_margin = 0.06
        right_margin = 0.03
        top_margin = 0.05
        bottom_margin = 0.06
        footer_offset = 0.020

    panel_w = columns_per_panel * cell_w
    fig_w = left_margin + num_panels * panel_w + (num_panels - 1) * panel_gap + right_margin
    fig_h = (
        top_margin
        + label_h
        + rows_per_panel * cell_h
        + max(rows_per_panel - 1, 0) * row_gap
        + footer_h
        + bottom_margin
    )
    fig = plt.figure(figsize=(fig_w, fig_h))

    panel_lefts = []
    x = left_margin / fig_w
    panel_width_norm = panel_w / fig_w
    panel_gap_norm = panel_gap / fig_w
    for panel_idx in range(num_panels):
        panel_lefts.append(x + panel_idx * (panel_width_norm + panel_gap_norm))

    content_bottom = bottom_margin / fig_h + footer_h / fig_h
    content_top = 1.0 - top_margin / fig_h - label_h / fig_h
    cell_w_norm = cell_w / fig_w
    cell_h_norm = cell_h / fig_h
    row_gap_norm = row_gap / fig_h

    axes: dict[tuple[int, int], plt.Axes] = {}
    for item_idx in range(num_items):
        panel_idx = item_idx // rows_per_panel
        row_in_panel = item_idx % rows_per_panel
        row_from_top = row_in_panel
        bottom = (
            content_top
            - (row_from_top + 1) * cell_h_norm
            - row_from_top * row_gap_norm
        )
        panel_left = panel_lefts[panel_idx]
        for col_idx in range(columns_per_panel):
            left = panel_left + col_idx * cell_w_norm
            ax = fig.add_axes([left, bottom, cell_w_norm, cell_h_norm])
            ax.axis("off")
            axes[(item_idx, col_idx)] = ax

    label_y = 1.0 - (top_margin + 0.58 * label_h) / fig_h
    footer_y = bottom_margin / fig_h + footer_offset
    return fig, axes, panel_lefts, panel_w / fig_w, label_y, footer_y


def main() -> None:
    args = parse_args()
    payload = load_results(args.results_json)
    stats = metric_map(payload)
    configure_matplotlib()

    indices = list(args.indices)
    source_prefix = resolve_source_prefix(args.images_dir)
    fig, axes, panel_lefts, panel_width, label_y, footer_y = create_layout(
        indices=indices,
        thumb_size=args.thumb_size,
        layout=args.layout,
    )

    num_items = len(indices)
    num_panels = 2 if num_items > 3 else 1
    rows_per_panel = int(np.ceil(num_items / num_panels))

    for item_idx, row_index in enumerate(indices):
        for col_idx, (column_key, _label) in enumerate(ALGORITHM_COLUMNS):
            ax = axes[(item_idx, col_idx)]
            path = image_path(
                args.images_dir,
                column_key=column_key,
                source_prefix=source_prefix,
                row_index=row_index,
            )
            ax.imshow(
                load_thumbnail(path, thumb_size=args.thumb_size),
                interpolation="lanczos",
                resample=True,
            )

    for panel_idx, panel_left in enumerate(panel_lefts):
        for col_idx, (column_key, label) in enumerate(ALGORITHM_COLUMNS):
            x = panel_left + ((col_idx + 0.5) / len(ALGORITHM_COLUMNS)) * panel_width
            fig.text(x, label_y, label, ha="center", va="center", fontsize=FS)
            fig.text(
                x,
                footer_y,
                build_footer_text(column_key, stats, layout=args.layout),
                ha="center",
                va="bottom",
                fontsize=FS,
                linespacing=0.95 if args.layout == "compact" else 1.0,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, pad_inches=0.01, dpi=300)
    plt.close(fig)
    print(f"Saved grid to {args.output}")


if __name__ == "__main__":
    main()
