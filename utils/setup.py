import json
import os
from typing import Any, Dict


def _default_config_path() -> str | None:
    # utils/setup.py -> repo root
    repo_root = os.path.dirname(os.path.dirname(__file__))
    candidate = os.path.join(repo_root, "config.yaml")
    return candidate if os.path.exists(candidate) else None


def load_config(config_path: str | None = None) -> Dict[str, Any]:
    """Load config.yaml and validate required fields. No extra normalization."""
    if not config_path:
        config_path = _default_config_path()
        if not config_path:
            raise FileNotFoundError("config.yaml not found at repo root")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("PyYAML is required to load config.yaml. Install 'pyyaml'.") from e

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("config.yaml root must be a mapping/object")

    if "global" not in data or not isinstance(data["global"], dict):
        raise ValueError("config.yaml must define a 'global' mapping")

    g = data["global"]
    missing = [k for k in ("run_name", "dataset", "metrics") if k not in g]
    if missing:
        raise ValueError(f"config.yaml: missing required global fields: {', '.join(missing)}")

    if "metrics" not in data or not isinstance(data["metrics"], dict):
        raise ValueError("config.yaml must define a 'metrics' mapping")

    return data