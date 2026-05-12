from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.io import mmread
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Table4DatasetSpec:
    key: str
    label: str
    leave_out: tuple[int, ...]
    default_times: tuple[float, ...] | None


TABLE4_DATASETS = {
    "eb": Table4DatasetSpec(
        key="eb",
        label="EB",
        leave_out=(1, 2, 3),
        default_times=None,
    ),
    "cite": Table4DatasetSpec(
        key="cite",
        label="Cite",
        leave_out=(1, 2),
        default_times=(2.0, 3.0, 4.0, 7.0),
    ),
    "multi": Table4DatasetSpec(
        key="multi",
        label="Multi",
        leave_out=(1, 2),
        default_times=(2.0, 3.0, 4.0, 7.0),
    ),
}
TABLE4_ORDER = ("eb", "cite", "multi")
DEFAULT_DONOR = 13176
DEFAULT_CITE_DAYS = (2, 3, 4, 7)
DEFAULT_MULTI_DAYS = (2, 3, 4, 7)
DEFAULT_EMBRYOID_TIMES = (0.0, 1.0, 2.0, 3.0, 4.0)
DEFAULT_PCA_EMBED_DIM = 100
EB_LIBRARY_SIZE_MIN = 1000.0
EB_LIBRARY_SIZE_MAX = 12000.0
EB_MITO_CUTOFF = 400.0
EB_RARE_GENE_MIN_CELLS = 10


def decode_bytes_array(values: np.ndarray) -> list[str]:
    return [
        value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ]


def finalize_embedding_timepoints(
    timepoints: list[np.ndarray],
    *,
    dims: int,
    whiten: bool,
) -> list[torch.Tensor]:
    lengths = [array.shape[0] for array in timepoints]
    stacked = np.concatenate(timepoints, axis=0).astype(np.float32, copy=False)
    if whiten:
        scaler = StandardScaler()
        stacked = scaler.fit_transform(stacked).astype(np.float32, copy=False)

    output_dims = min(dims, stacked.shape[1])
    stacked = stacked[:, :output_dims].astype(np.float32, copy=False)

    outputs: list[torch.Tensor] = []
    start = 0
    for length in lengths:
        outputs.append(torch.from_numpy(stacked[start : start + length].copy()))
        start += length
    return outputs


def iter_sparse_row_batches(matrix: sp.csr_matrix, batch_size: int):
    for start in range(0, matrix.shape[0], batch_size):
        batch = matrix[start : start + batch_size].toarray().astype(
            np.float32,
            copy=False,
        )
        yield batch


def read_h5_axis1_ids(path: Path) -> list[str]:
    import h5py
    import hdf5plugin  # noqa: F401

    with h5py.File(path, "r") as handle:
        group = handle[list(handle.keys())[0]]
        return decode_bytes_array(group["axis1"][:])


def iterate_h5_row_batches(
    path: Path,
    row_indices: np.ndarray,
    *,
    batch_size: int,
):
    import h5py
    import hdf5plugin  # noqa: F401

    with h5py.File(path, "r") as handle:
        group = handle[list(handle.keys())[0]]
        values = group["block0_values"]
        sorted_indices = np.sort(np.asarray(row_indices, dtype=np.int64))
        for start in range(0, len(sorted_indices), batch_size):
            batch_indices = sorted_indices[start : start + batch_size]
            yield np.asarray(values[batch_indices, :], dtype=np.float32)


def fit_incremental_pca_from_batches(
    batch_iterators: list[tuple[Path, np.ndarray]],
    *,
    requested_components: int,
    batch_size: int,
) -> IncrementalPCA:
    ipca: IncrementalPCA | None = None
    for path, row_indices in batch_iterators:
        for batch in iterate_h5_row_batches(path, row_indices, batch_size=batch_size):
            if ipca is None:
                n_components = max(
                    1,
                    min(
                        requested_components,
                        batch.shape[0],
                        batch.shape[1],
                        batch_size,
                    ),
                )
                ipca = IncrementalPCA(
                    n_components=n_components,
                    batch_size=batch_size,
                )
            if batch.shape[0] < ipca.n_components:
                continue
            ipca.partial_fit(batch)
    if ipca is None:
        raise ValueError("Failed to fit IncrementalPCA: no batches were available.")
    return ipca


def transform_h5_timepoints(
    file_day_indices: dict[Path, dict[int, np.ndarray]],
    *,
    ipca: IncrementalPCA | None,
    raw_dim: int,
    batch_size: int,
) -> dict[int, np.ndarray]:
    transformed: dict[int, list[np.ndarray]] = {}
    for path, day_map in file_day_indices.items():
        for day, row_indices in day_map.items():
            outputs = []
            for batch in iterate_h5_row_batches(path, row_indices, batch_size=batch_size):
                if ipca is not None:
                    batch = ipca.transform(batch).astype(np.float32, copy=False)
                else:
                    batch = batch[:, :raw_dim].astype(np.float32, copy=False)
                outputs.append(batch)
            transformed.setdefault(day, []).extend(outputs)
    return {
        day: np.concatenate(batches, axis=0) for day, batches in transformed.items()
    }


def fit_incremental_pca_from_sparse_matrix(
    matrix: sp.csr_matrix,
    *,
    requested_components: int,
    batch_size: int,
) -> IncrementalPCA:
    ipca: IncrementalPCA | None = None
    for batch in iter_sparse_row_batches(matrix, batch_size):
        if ipca is None:
            n_components = max(
                1,
                min(
                    requested_components,
                    batch.shape[0],
                    batch.shape[1],
                    batch_size,
                ),
            )
            ipca = IncrementalPCA(
                n_components=n_components,
                batch_size=batch_size,
            )
        if batch.shape[0] < ipca.n_components:
            continue
        ipca.partial_fit(batch)
    if ipca is None:
        raise ValueError("Failed to fit IncrementalPCA: no sparse batches were available.")
    return ipca


def transform_sparse_matrix(
    matrix: sp.csr_matrix,
    *,
    ipca: IncrementalPCA | None,
    raw_dim: int,
    batch_size: int,
) -> np.ndarray:
    outputs = []
    for batch in iter_sparse_row_batches(matrix, batch_size):
        if ipca is not None:
            batch = ipca.transform(batch).astype(np.float32, copy=False)
        else:
            batch = batch[:, :raw_dim].astype(np.float32, copy=False)
        outputs.append(batch)
    return np.concatenate(outputs, axis=0)


def prepare_cite_or_multi_timepoints(
    spec: Table4DatasetSpec,
    *,
    data_root: Path,
    dims: int,
    pca_embed_dim: int,
    fit_pca: bool,
    whiten: bool,
    pca_batch_size: int,
    donor: int,
) -> tuple[list[torch.Tensor], np.ndarray, str]:
    cite_multi_root = data_root / "cite_multi"
    metadata = pd.read_csv(cite_multi_root / "metadata.csv")
    if spec.key == "cite":
        technology = "citeseq"
        days = DEFAULT_CITE_DAYS
        matrix_files = (
            cite_multi_root / "train_cite_inputs.h5",
            cite_multi_root / "test_cite_inputs.h5",
        )
    else:
        technology = "multiome"
        days = DEFAULT_MULTI_DAYS
        matrix_files = (cite_multi_root / "train_multi_targets.h5",)

    subset = metadata[
        (metadata["technology"] == technology)
        & (metadata["donor"] == donor)
        & (metadata["day"].isin(days))
    ][["cell_id", "day"]].copy()
    if subset.empty:
        raise ValueError(f"{spec.label}: no cells found for donor {donor}.")

    file_day_indices: dict[Path, dict[int, np.ndarray]] = {}
    unresolved = set(subset["cell_id"].tolist())
    for path in matrix_files:
        cell_ids = read_h5_axis1_ids(path)
        id_to_row = {cell_id: idx for idx, cell_id in enumerate(cell_ids)}
        rows_by_day: dict[int, list[int]] = {}
        for day in days:
            day_ids = subset.loc[subset["day"] == day, "cell_id"]
            found_ids = [cell_id for cell_id in day_ids if cell_id in id_to_row]
            rows = [id_to_row[cell_id] for cell_id in found_ids]
            if rows:
                rows_by_day[int(day)] = rows
                unresolved.difference_update(found_ids)
        if rows_by_day:
            file_day_indices[path] = {
                day: np.asarray(sorted(rows), dtype=np.int64)
                for day, rows in rows_by_day.items()
            }

    if unresolved:
        raise ValueError(
            f"{spec.label}: failed to locate {len(unresolved)} cells in the H5 files. "
            f"Example missing cell ids: {sorted(list(unresolved))[:5]}"
        )

    fit_inputs = []
    for path, day_map in file_day_indices.items():
        fit_indices = np.concatenate(list(day_map.values()))
        fit_inputs.append((path, np.unique(np.sort(fit_indices))))
    ipca = (
        fit_incremental_pca_from_batches(
            fit_inputs,
            requested_components=max(dims, pca_embed_dim),
            batch_size=pca_batch_size,
        )
        if fit_pca
        else None
    )
    transformed = transform_h5_timepoints(
        file_day_indices,
        ipca=ipca,
        raw_dim=dims,
        batch_size=pca_batch_size,
    )
    ordered_days = [day for day in days if day in transformed]
    if ordered_days != list(days):
        raise ValueError(f"{spec.label}: expected days {days}, found {ordered_days}.")
    raw_timepoints = [transformed[day] for day in ordered_days]
    artifact_desc = (
        f"{cite_multi_root.name}:{technology}:donor={donor}:days={','.join(map(str, ordered_days))}"
        f":raw->pca{max(dims, pca_embed_dim) if fit_pca else 'none'}"
        f"->whiten={whiten}->dims={dims}"
    )
    return (
        finalize_embedding_timepoints(raw_timepoints, dims=dims, whiten=whiten),
        np.asarray(ordered_days, dtype=np.float32),
        artifact_desc,
    )


def load_embryoid_matrix(timepoint_dir: Path):
    return mmread(timepoint_dir / "matrix.mtx").tocsr().transpose().tocsr()


def load_embryoid_gene_symbols(timepoint_dir: Path) -> np.ndarray:
    genes = pd.read_csv(timepoint_dir / "genes.tsv", sep="\t", header=None)
    if genes.shape[1] >= 2:
        return genes.iloc[:, 1].astype(str).to_numpy()
    return genes.iloc[:, 0].astype(str).to_numpy()


def filter_sparse_rows_by_library_size(
    matrix: sp.csr_matrix,
    *,
    min_total: float,
    max_total: float,
) -> tuple[sp.csr_matrix, np.ndarray]:
    totals = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float32, copy=False)
    keep = (totals >= min_total) & (totals <= max_total)
    return matrix[keep].tocsr(), keep


def filter_sparse_matrix_by_mito_expression(
    matrix: sp.csr_matrix,
    labels: np.ndarray,
    gene_symbols: np.ndarray,
    *,
    cutoff: float,
    target_sum: float,
) -> tuple[sp.csr_matrix, np.ndarray]:
    mito_mask = np.char.startswith(np.char.upper(gene_symbols.astype(str)), "MT-")
    if not np.any(mito_mask):
        return matrix, labels

    totals = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float32, copy=False)
    totals = np.clip(totals, 1e-6, None)
    mito_totals = np.asarray(matrix[:, mito_mask].sum(axis=1)).ravel().astype(
        np.float32,
        copy=False,
    )
    mito_expression = mito_totals / totals * target_sum
    keep = mito_expression <= cutoff
    return matrix[keep].tocsr(), labels[keep]


def filter_sparse_matrix_by_rare_genes(
    matrix: sp.csr_matrix,
    gene_symbols: np.ndarray,
    *,
    min_cells: int,
) -> tuple[sp.csr_matrix, np.ndarray]:
    keep = matrix.getnnz(axis=0) >= min_cells
    return matrix[:, keep].tocsr(), gene_symbols[keep]


def library_size_normalize_sparse(
    matrix: sp.csr_matrix,
    *,
    target_sum: float,
) -> sp.csr_matrix:
    totals = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float32, copy=False)
    scales = target_sum / np.clip(totals, 1e-6, None)
    return matrix.multiply(scales[:, None]).tocsr()


def sqrt_transform_sparse(matrix: sp.csr_matrix) -> sp.csr_matrix:
    matrix = matrix.astype(np.float32, copy=True).tocsr()
    np.sqrt(matrix.data, out=matrix.data)
    return matrix


def prepare_embryoid_timepoints(
    *,
    data_root: Path,
    dims: int,
    pca_embed_dim: int,
    fit_pca: bool,
    whiten: bool,
    pca_batch_size: int,
) -> tuple[list[torch.Tensor], np.ndarray, str]:
    embryoid_root = data_root / "embryoid" / "scRNAseq"
    if not embryoid_root.exists():
        raise FileNotFoundError(f"Missing embryoid directory: {embryoid_root}")

    timepoint_dirs = sorted(
        [path for path in embryoid_root.iterdir() if path.is_dir()],
        key=lambda path: [
            int(part) if part.isdigit() else part
            for part in re.split(r"(\d+)", path.name)
        ],
    )
    if len(timepoint_dirs) != 5:
        raise ValueError(
            f"Expected 5 embryoid timepoint folders, found {len(timepoint_dirs)}."
        )

    gene_symbols: np.ndarray | None = None
    filtered_batches: list[sp.csr_matrix] = []
    batch_labels: list[np.ndarray] = []

    for index, path in enumerate(timepoint_dirs):
        matrix = load_embryoid_matrix(path)
        current_genes = load_embryoid_gene_symbols(path)
        if gene_symbols is None:
            gene_symbols = current_genes
        elif not np.array_equal(gene_symbols, current_genes):
            raise ValueError(f"Embryoid gene order mismatch in {path}.")

        matrix, keep = filter_sparse_rows_by_library_size(
            matrix,
            min_total=EB_LIBRARY_SIZE_MIN,
            max_total=EB_LIBRARY_SIZE_MAX,
        )
        filtered_batches.append(matrix)
        batch_labels.append(np.full(int(keep.sum()), index, dtype=np.int64))

    if gene_symbols is None:
        raise ValueError("Embryoid preprocessing failed: no genes were loaded.")

    combined = sp.vstack(filtered_batches).tocsr()
    labels = np.concatenate(batch_labels, axis=0)
    combined, labels = filter_sparse_matrix_by_mito_expression(
        combined,
        labels,
        gene_symbols,
        cutoff=EB_MITO_CUTOFF,
        target_sum=10_000.0,
    )
    combined, gene_symbols = filter_sparse_matrix_by_rare_genes(
        combined,
        gene_symbols,
        min_cells=EB_RARE_GENE_MIN_CELLS,
    )
    combined = library_size_normalize_sparse(combined, target_sum=10_000.0)
    combined = sqrt_transform_sparse(combined)

    ipca = (
        fit_incremental_pca_from_sparse_matrix(
            combined,
            requested_components=max(dims, pca_embed_dim),
            batch_size=pca_batch_size,
        )
        if fit_pca
        else None
    )
    embedding = transform_sparse_matrix(
        combined,
        ipca=ipca,
        raw_dim=dims,
        batch_size=pca_batch_size,
    )
    raw_timepoints = [embedding[labels == index] for index in range(len(timepoint_dirs))]
    for index, array in enumerate(raw_timepoints):
        if array.shape[0] == 0:
            raise ValueError(
                f"Embryoid preprocessing removed all cells at timepoint index {index}."
            )

    return (
        finalize_embedding_timepoints(raw_timepoints, dims=dims, whiten=whiten),
        np.asarray(DEFAULT_EMBRYOID_TIMES, dtype=np.float32),
        (
            f"{embryoid_root}:raw->library[{int(EB_LIBRARY_SIZE_MIN)},{int(EB_LIBRARY_SIZE_MAX)}]"
            f"->mito<={int(EB_MITO_CUTOFF)}->rare>={EB_RARE_GENE_MIN_CELLS}"
            f"->normalize=1e4->sqrt->pca{max(dims, pca_embed_dim) if fit_pca else 'none'}"
            f"->whiten={whiten}->dims={dims}"
        ),
    )


def load_real_dataset(
    spec: Table4DatasetSpec,
    *,
    data_root: Path,
    dims: int,
    pca_embed_dim: int,
    fit_pca: bool,
    whiten: bool,
    pca_batch_size: int,
    donor: int,
) -> tuple[list[torch.Tensor], np.ndarray, str]:
    if spec.key == "eb":
        return prepare_embryoid_timepoints(
            data_root=data_root,
            dims=dims,
            pca_embed_dim=pca_embed_dim,
            fit_pca=fit_pca,
            whiten=whiten,
            pca_batch_size=pca_batch_size,
        )
    return prepare_cite_or_multi_timepoints(
        spec,
        data_root=data_root,
        dims=dims,
        pca_embed_dim=pca_embed_dim,
        fit_pca=fit_pca,
        whiten=whiten,
        pca_batch_size=pca_batch_size,
        donor=donor,
    )
