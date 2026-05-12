set -ex
git pull
uv sync
PYTHONPATH=experiments_2d ./.venv/bin/python -u experiments_2d/main.py \
    --quality-eval-every 1 \
    --output-json results/output_toy_2d_suite.json \
    | tee results/output_toy_2d_suite.txt
