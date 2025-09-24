
#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

uv run python3 -m endpoints.run_models "/fs/clip-quiz/nbalepur/mcqa-bench/local_datasets/ARC/train.csv" \
--run-name "my_run" \
--models openai/gpt-5-nano-2025-08-07 openai/gpt-5-mini-2025-08-07 openai/gpt-5-2025-08-07 \
--limit 3 \
--parallel