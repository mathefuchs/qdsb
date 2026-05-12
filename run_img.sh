set -ex
git pull
uv sync
PYTHONPATH=experiments_img ./.venv/bin/python -u experiments_img/main.py \
    --device cuda --output-json results/output_img.json | tee results/output_img.txt
