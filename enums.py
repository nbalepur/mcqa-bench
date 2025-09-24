from enum import Enum
import importlib
from typing import Callable, Dict, List
from inspect_ai.scorer import Scorer


class Metrics(Enum):
    DIFFICULTY = "difficulty"
    SHORTCUTS = "shortcuts"
    CONTAMINATION = "contamination"
    WRITING_FLAWS = "writing_flaws"

    @classmethod
    def values(cls) -> list[str]:
        return [m.value for m in cls]


# Map metric name -> "module:function" for scorer factories
METRIC_SCORER_SPEC: Dict[str, str] = {
    Metrics.DIFFICULTY.value: "scorers.difficulty_scorer:difficulty_scorer",
    Metrics.SHORTCUTS.value: "scorers.shortcut_scorer:shortcut_scorer",
    Metrics.CONTAMINATION.value: "scorers.contamination_scorer:contamination_scorer",
    Metrics.WRITING_FLAWS.value: "scorers.writing_flaws_scorer:writing_flaws_scorer",
}


def get_metric_scorer_factory(metric: str) -> Callable[[], Scorer]:
    spec = METRIC_SCORER_SPEC.get(metric)
    if not spec:
        raise KeyError(f"No scorer mapping for metric '{metric}'")
    module_name, func_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, func_name)
    return factory


def get_scorers_for_metrics(metrics: List[str]) -> List[Scorer]:
    scorers: List[Scorer] = []
    for metric in metrics:
        factory = get_metric_scorer_factory(metric)
        scorers.append(factory())
    return scorers


