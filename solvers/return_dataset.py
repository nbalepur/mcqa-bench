from inspect_ai.solver import TaskState
from inspect_ai.solver import Generate
from inspect_ai.solver import solver


@solver
def return_dataset():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.metadata['question'] = state.user_prompt.text
        state.metadata['choices'] = [c.value for c in state.choices]
        state.metadata['target'] = state.target.text
        return state
    return solve