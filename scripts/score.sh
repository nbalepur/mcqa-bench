export PYTHONPATH="${PYTHONPATH}:$(pwd)"

uv run inspect eval endpoints/score_dataset.py@mcqa_dataset_score --limit 5