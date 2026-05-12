# QDSB: Quantized Diffusion Schrödinger Bridges

This repository contains the experimental code accompanying the paper *QDSB: Quantized Diffusion Schrödinger Bridges*. It includes the real-world cell biology experiments, 2D toy experiments, image experiments, plotting utilities, and result extraction scripts used to produce the paper figures and tables.

## Overview

The repository is organized around three experiment suites:

- `experiments/`: single-cell benchmarks on EB, Cite, and Multi.
- `experiments_2d/`: 2D toy benchmarks and sensitivity studies.
- `experiments_img/`: unpaired FFHQ latent-space image experiments.

The repository also contains:

- `results/`: JSON/TXT outputs and scripts for extracting plots and tables.
- `plots/`: generated paper figures.
- top-level `run_*.sh` scripts: convenience entry points for the main experiments.

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/).

1. Install `uv`.
2. Create and sync the environment:

```bash
uv sync
```

3. Activate the environment or use the local interpreter directly:

```bash
source .venv/bin/activate
```

The project currently targets Python `>=3.14`; see [`pyproject.toml`](pyproject.toml).

## Reproducing the experiments

### Real-world cell biology experiments

The cell experiments live in [`experiments/`](experiments/) and can be launched via the top-level scripts:

- [`run_qdsb.sh`](run_qdsb.sh)
- [`run_dsb_cell.sh`](run_dsb_cell.sh)
- [`run_dsbm_cell.sh`](run_dsbm_cell.sh)
- [`run_sf2m_cell.sh`](run_sf2m_cell.sh)
- [`run_sf2m_mpot_cell.sh`](run_sf2m_mpot_cell.sh)
- [`run_lightsb_m_cell.sh`](run_lightsb_m_cell.sh)

Each script writes a JSON file and a matching text log to [`results/`](results/).

### 2D toy experiments

The 2D experiments are centered around [`experiments_2d/main.py`](experiments_2d/main.py). The main benchmark is launched with:

```bash
./run_toy_2d_suite.sh
```

This produces [`results/output_toy_2d_suite.json`](results/output_toy_2d_suite.json) and the corresponding text log.

The QDSB anchor sensitivity study is implemented in [`experiments_2d/qdsb_anchor_sensitivity.py`](experiments_2d/qdsb_anchor_sensitivity.py).

### Image experiments

The image experiments are implemented in [`experiments_img/main.py`](experiments_img/main.py) and are launched with:

```bash
./run_img.sh
```

By default, this runs the unpaired FFHQ latent-space experiments with a fixed training-time budget and writes:

- [`results/output_img.json`](results/output_img.json)
- [`results/output_img.txt`](results/output_img.txt)

The image pipeline requires the FFHQ ALAE checkpoint at: `ALAE/training_artifacts/ffhq/model_157.pth`.
This checkpoint is about `600 MB` and is not stored in this repository. It can be downloaded from <https://alaeweights.s3.us-east-2.amazonaws.com/ffhq/model_157.pth>.

The image experiments use the adapted ALAE inference code in [`ALAE/`](ALAE/). The dataset is prepared automatically into `data/ffhq` if it is not already present.

## Plots and tables

The scripts in [`results/`](results/) convert stored JSON outputs into the paper plots and tables.

- [`plot_time_vs_mmd.py`](results/plot_time_vs_mmd.py)
- [`plot_time_vs_mmd_toy.py`](results/plot_time_vs_mmd_toy.py)
- [`plot_qdsb_anchor_sensitivity.py`](results/plot_qdsb_anchor_sensitivity.py)
- [`plot_img_grid.py`](results/plot_img_grid.py)
- [`extract_mmd_table_at_time.py`](results/extract_mmd_table_at_time.py)

Generated figures are stored in [`plots/`](plots/).

## Repository structure

### Included in the artifact

- [`ALAE/`](ALAE/): minimal subset of the ALAE codebase needed for FFHQ latent encoding/decoding in the image experiments. The original ALAE work is by Stanislav Pidhorskyi, Donald A. Adjeroh, and Gianfranco Doretto, *Adversarial Latent Autoencoders* (CVPR 2020). We adapted their code for this repository. The original ALAE repository does not ship a standalone `LICENSE` file; the retained source files keep their upstream license headers.
- [`experiments/`](experiments/): real-world cellular biology experiments.
- [`experiments_2d/`](experiments_2d/): 2D toy experiments and sensitivity studies.
- [`experiments_img/`](experiments_img/): image experiments.
- [`plots/`](plots/): generated plots used in the paper.
- [`results/`](results/): detailed JSON/TXT outputs and plotting/table scripts.
- [`pyproject.toml`](pyproject.toml): `uv` project configuration.
- [`README.md`](README.md): this file.
- `run_*.sh`: top-level scripts for running the main experiments.
- [`.gitignore`](.gitignore)
- [`.python-version`](.python-version)

Each experiment suite contains a [`torchcfm/`](experiments/torchcfm/) module derived from the SF2M / conditional-flow-matching codebase by Tong et al. The original repository is: <https://github.com/atong01/conditional-flow-matching>.

Local copies of the upstream MIT license are included in:

- [`experiments/torchcfm/LICENSE`](experiments/torchcfm/LICENSE)
- [`experiments_2d/torchcfm/LICENSE`](experiments_2d/torchcfm/LICENSE)
- [`experiments_img/torchcfm/LICENSE`](experiments_img/torchcfm/LICENSE)

## Notes

- The main outputs can be reproduced from the scripts in `run_*.sh` and the post-processing utilities in `results/`.
- The repository contains generated result files already, so figures and tables can also be recreated directly from the stored JSON outputs without rerunning every experiment.
- All experiments were conducted on AMD Ryzen 9 7900X 12-Core, NVIDIA GeForce RTX 4090.
