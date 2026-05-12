set -ex
git pull
uv sync
./.venv/bin/python -u experiments/dsb_cell.py \
    --output-json results/output_mmd_dsb_cell.json \
    --quality-eval-every 1 --epochs 1000 | tee results/output_mmd_dsb_cell.txt
