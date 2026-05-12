set -ex
git pull
uv sync
./.venv/bin/python -u experiments/sf2m_cell.py \
    --output-json results/output_mmd_sf2m_cell.json \
    --quality-eval-every 1 --epochs 1000 | tee results/output_mmd_sf2m_cell.txt
