from enum import Enum
from typing import Dict, List, Any
from inspect_ai.scorer import Scorer

from scorers.difficulty_scorer import accuracy_scorer, difficulty_scorer, discriminability_scorer
from scorers.shortcut_scorer import shortcut_scorer
from scorers.contamination_scorer import contamination_scorer
from scorers.writing_flaws_scorer import writing_flaws_scorer


class Metrics(Enum):
    DIFFICULTY = "difficulty"
    SHORTCUTS = "shortcuts"
    CONTAMINATION = "contamination"
    WRITING_FLAWS = "writing_flaws"

    @classmethod
    def values(cls) -> list[str]:
        return [m.value for m in cls]


def get_scorer_for_metric(metric: str, config: Dict[str, Any] | None = None, **kwargs) -> Scorer | List[Scorer]:
    """Get scorer for a metric with optional config parameters."""
    from utils.setup import load_config
    config = load_config()
    
    metrics_config = config.get("metrics", {})
    metric_config = metrics_config.get(metric, {})
    
    if metric == Metrics.DIFFICULTY.value:
        return [accuracy_scorer(
            models=metric_config.get("models", []),
            use_cot=metric_config.get("use_cot", False),
            sample_to_score=kwargs.get('sample_to_score')
        ), difficulty_scorer(
            sample_to_score=kwargs.get('sample_to_score')
        ),
        discriminability_scorer(
            sample_to_score=kwargs.get('sample_to_score')
        )]
    elif metric == Metrics.SHORTCUTS.value:
        return shortcut_scorer(
            model=metric_config.get("model", "openai/gpt-4o-mini"),
            use_cot=metric_config.get("use_cot", False),
            num_shuffles=metric_config.get("num_shuffles", 3),
            correction_strategy=metric_config.get("correction_strategy", "none")
        )
    elif metric == Metrics.CONTAMINATION.value:
        return contamination_scorer(
            model=metric_config.get("model", "openai/gpt-4o-mini"),
            use_cot=metric_config.get("use_cot", False),
            correction_strategy=metric_config.get("correction_strategy", "none")
        )
    elif metric == Metrics.WRITING_FLAWS.value:
        return writing_flaws_scorer(
            model=metric_config.get("model", "openai/gpt-4o-mini"),
            use_cot=metric_config.get("use_cot", False),
            num_shuffles=metric_config.get("num_shuffles", 3),
            correction_strategy=metric_config.get("correction_strategy", "none")
        )
    else:
        valid_metrics = [m.value for m in Metrics]
        raise KeyError(f"No scorer mapping for metric '{metric}'. Valid metrics: {valid_metrics}")


def get_scorers_for_metrics(metrics: List[str], config: Dict[str, Any] | None = None, save_skills: bool = False, **kwargs) -> List[Scorer]:
    return [get_scorer_for_metric(metric, config, save_skills, **kwargs) for metric in metrics]

