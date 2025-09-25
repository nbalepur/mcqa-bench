#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

uv run python3 -m endpoints.run_skills --limit 20