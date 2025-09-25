#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

uv run python3 -c "
from endpoints.run_metrics import run_metrics_eval
run_metrics_eval()
"
