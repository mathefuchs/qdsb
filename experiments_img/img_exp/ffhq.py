from __future__ import annotations

import csv
import os
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_FFHQ_ROOT = Path("data/ffhq")
DEFAULT_FFHQ_HF_REPO_ID = "NUS-SRI-2025/FFHQ-Aging-Dataset"
DEFAULT_FFHQ_CHILD_AGE_GROUPS = ("0-2", "3-6", "7-9")
DEFAULT_FFHQ_ADULT_AGE_GROUPS = ("20-29", "30-39", "40-49")
DEFAULT_FFHQ_EVAL_FRACTION = 0.1
DEFAULT_FFHQ_SPLIT_SEED = 0


@dataclass(frozen=True)
class PreparedFFHQDirs:
    source_train: Path
    target_train: Path
    source_eval: Path
    target_eval: Path
    image_root: Path
    labels_csv: Path


def default_ffhq_dirs(root: Path) -> PreparedFFHQDirs:
    return PreparedFFHQDirs(
        source_train=root / "adult" / "train",
        target_train=root / "child" / "train",
        source_eval=root / "adult" / "eval",
        target_eval=root / "child" / "eval",
        image_root=root / "images1024x1024",
        labels_csv=root / "labels" / "ffhq_aging_labels.csv",
    )


def _dir_has_images(path: Path) -> bool:
    return path.exists() and any(child.is_file() for child in path.iterdir())


def prepared_ffhq_dirs_exist(root: Path) -> bool:
    dirs = default_ffhq_dirs(root)
    return all(
        _dir_has_images(path)
        for path in (dirs.source_train, dirs.target_train, dirs.source_eval, dirs.target_eval)
    )


def _download_ffhq_aging_dataset(root: Path, *, repo_id: str) -> Path:
    cache_dir = root / "_hf_ffhq_aging"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=cache_dir,
            allow_patterns=[
                "labels/ffhq_aging_labels.csv",
                "images_zip/images1024x1024/*.zip",
                "README.md",
            ],
        )
    )


def _extract_image_archives(download_dir: Path, *, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_dir = download_dir / "images_zip" / "images1024x1024"
    archives = sorted(zip_dir.glob("*.zip"))
    if not archives:
        raise FileNotFoundError(f"No FFHQ image archives found in {zip_dir}")
    marker = output_dir / ".extracted_complete"
    if marker.exists():
        return
    for archive in archives:
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(output_dir)
    marker.write_text("ok\n")


def _copy_labels_csv(download_dir: Path, *, output_path: Path) -> None:
    source = download_dir / "labels" / "ffhq_aging_labels.csv"
    if not source.exists():
        raise FileNotFoundError(f"FFHQ-Aging labels CSV not found at {source}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(source.read_bytes())


def _parse_age_groups(labels_csv: Path) -> tuple[list[int], list[str]]:
    image_numbers: list[int] = []
    age_groups: list[str] = []
    with labels_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if "image_number" not in reader.fieldnames or "age_group" not in reader.fieldnames:
            raise ValueError(
                f"{labels_csv} must contain image_number and age_group columns. "
                f"Found {reader.fieldnames}."
            )
        for row in reader:
            image_numbers.append(int(row["image_number"]))
            age_groups.append(str(row["age_group"]).strip())
    return image_numbers, age_groups


def _resolve_image_path(image_root: Path, image_number: int) -> Path:
    candidates = (
        image_root / f"{image_number:05d}.png",
        image_root / f"{image_number}.png",
        image_root / f"{image_number:05d}.jpg",
        image_root / f"{image_number}.jpg",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Unable to locate FFHQ image for image_number={image_number} in {image_root}")


def _reset_directory(path: Path) -> None:
    if path.exists():
        for child in path.iterdir():
            if child.is_symlink() or child.is_file():
                child.unlink()
            elif child.is_dir():
                _reset_directory(child)
                child.rmdir()
    path.mkdir(parents=True, exist_ok=True)


def _link_file(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        rel = os.path.relpath(source, destination.parent)
        destination.symlink_to(rel)
    except OSError:
        os.link(source, destination)


def _build_split(
    *,
    image_numbers: list[int],
    age_groups: list[str],
    selected_age_groups: tuple[str, ...],
    image_root: Path,
    train_dir: Path,
    eval_dir: Path,
    eval_fraction: float,
    split_seed: int,
) -> None:
    filtered = [num for num, age_group in zip(image_numbers, age_groups, strict=True) if age_group in selected_age_groups]
    if not filtered:
        raise ValueError(f"No FFHQ images found for age groups {selected_age_groups}.")
    rng = random.Random(split_seed)
    filtered = list(filtered)
    rng.shuffle(filtered)
    eval_count = max(1, int(round(len(filtered) * eval_fraction)))
    eval_ids = set(filtered[:eval_count])
    train_ids = filtered[eval_count:]
    if not train_ids:
        raise ValueError("Evaluation split consumed the entire FFHQ subset; reduce eval_fraction.")

    _reset_directory(train_dir)
    _reset_directory(eval_dir)
    for image_number in train_ids:
        source = _resolve_image_path(image_root, image_number)
        _link_file(source, train_dir / source.name)
    for image_number in sorted(eval_ids):
        source = _resolve_image_path(image_root, image_number)
        _link_file(source, eval_dir / source.name)


def ensure_prepared_ffhq_dirs(
    *,
    root: Path = DEFAULT_FFHQ_ROOT,
    repo_id: str = DEFAULT_FFHQ_HF_REPO_ID,
    child_age_groups: tuple[str, ...] = DEFAULT_FFHQ_CHILD_AGE_GROUPS,
    adult_age_groups: tuple[str, ...] = DEFAULT_FFHQ_ADULT_AGE_GROUPS,
    eval_fraction: float = DEFAULT_FFHQ_EVAL_FRACTION,
    split_seed: int = DEFAULT_FFHQ_SPLIT_SEED,
) -> PreparedFFHQDirs:
    dirs = default_ffhq_dirs(root)
    if prepared_ffhq_dirs_exist(root):
        return dirs

    root.mkdir(parents=True, exist_ok=True)
    download_dir = _download_ffhq_aging_dataset(root, repo_id=repo_id)
    _copy_labels_csv(download_dir, output_path=dirs.labels_csv)
    _extract_image_archives(download_dir, output_dir=dirs.image_root)
    image_numbers, age_groups = _parse_age_groups(dirs.labels_csv)
    _build_split(
        image_numbers=image_numbers,
        age_groups=age_groups,
        selected_age_groups=adult_age_groups,
        image_root=dirs.image_root,
        train_dir=dirs.source_train,
        eval_dir=dirs.source_eval,
        eval_fraction=eval_fraction,
        split_seed=split_seed,
    )
    _build_split(
        image_numbers=image_numbers,
        age_groups=age_groups,
        selected_age_groups=child_age_groups,
        image_root=dirs.image_root,
        train_dir=dirs.target_train,
        eval_dir=dirs.target_eval,
        eval_fraction=eval_fraction,
        split_seed=split_seed,
    )
    return dirs
