import os
import random
import string
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState
from inspect_ai.scorer import scorer, Score, mean, stderr


def _random_explanation(length: int = 24) -> str:
    letters = string.ascii_lowercase
    token = "".join(random.choice(letters) for _ in range(length))
    return f"random-expl:{token}"


@scorer(metrics=[mean(), stderr()])
def shortcut_scorer(model: str = "openai/gpt-4o-mini", use_cot: bool = False, num_shuffles: int = 3, correction_strategy: str = "none"):
    async def _score(state: TaskState, target: Target) -> Score:
        value = random.random()
        explanation = _random_explanation()
        return Score(value=value, explanation=explanation)

    return _score

