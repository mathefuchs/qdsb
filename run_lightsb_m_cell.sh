set -ex
git pull
uv sync
./.venv/bin/python -u experiments/lightsb_m_cell.py \
    --output-json results/output_mmd_lightsb_m_cell.json \
    --quality-eval-every 1 --epochs 1000 | tee results/output_mmd_lightsb_m_cell.txt
