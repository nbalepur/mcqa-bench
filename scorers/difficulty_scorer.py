import copy
import string
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState
from inspect_ai.solver._multiple_choice import parse_answers, set_choices_based_on_generated_response
from inspect_ai.model import get_model
from inspect_ai.scorer import scorer, Score, mean, stderr, choice
from inspect_ai.scorer._metric import CORRECT

from prompts.run_mcqa_prompts import load_mcqa_template





@scorer(name="accuracy", metrics=[mean(), stderr()])
def accuracy_scorer(models: list = None, use_cot: bool = False, sample_to_score: dict = None):

    async def _score(state: TaskState, target: Target) -> Score:
        # If prior logs are provided, use cached scores
        if sample_to_score:
            return Score(
                value=sum([float(s["score"] == CORRECT) for s in sample_to_score[state.sample_id]['accuracy']]) / len(sample_to_score[state.sample_id]['accuracy']),
                explanation="",
                metadata={'scores': sample_to_score[state.sample_id]['accuracy']},
            )
        
        # Otherwise, generate new model outputs
        total_score = 0.0
        meta_data = []
        
        for model in models:
            mcqa_model = get_model(model)
            template = load_mcqa_template(use_cot, with_question=True)
            
            score_prompt = template.format(
                question=state.metadata['question'],
                choices=state.metadata['choices'],
                letters=",".join(string.ascii_uppercase[:len(state.choices)]),
            )
            output = await mcqa_model.generate(score_prompt)
            
            # Create a copy of the state with the model's output
            temp_state = TaskState(
                model=state.model,
                sample_id=state.sample_id,
                epoch=state.epoch,
                input=state.input,
                messages=state.messages,
                choices=copy.deepcopy(state.choices),
                output=output,
                metadata=state.metadata
            )
            answers = parse_answers(temp_state, multiple_correct=False)
            set_choices_based_on_generated_response(temp_state, answers)
            
            score = await choice()(temp_state, target)
            
            meta_data.append({
                'model': model,
                'score': score.value,
                'completion': output.completion,
                'sample_id': state.sample_id
            })
            total_score += int(score.value == CORRECT)
        
        meta_data = {'scores': meta_data}
        state.metadata = state.metadata | meta_data
        return Score(
            value=total_score / len(models), 
            explanation=f"",
            metadata=meta_data,
        )

    return _score


@scorer(name="diff", metrics=[mean(), stderr()])
def difficulty_scorer(sample_to_score: dict = None):
    """Scorer that returns IRT difficulty scores for each question."""
    
    async def _score(state: TaskState, target: Target) -> Score:
        if not sample_to_score:
            return Score(value=0.0, explanation="No IRT values provided")        
        return Score(
            value=sample_to_score[state.sample_id]['difficulty'],
            explanation="",
        )
    
    return _score


@scorer(name="disc", metrics=[mean(), stderr()])
def discriminability_scorer(sample_to_score: dict = None):
    """Scorer that returns IRT discriminability scores for each question."""
    
    async def _score(state: TaskState, target: Target) -> Score:
        if not sample_to_score:
            return Score(value=0.0, explanation="No IRT values provided")        
        return Score(
            value=sample_to_score[state.sample_id]['discriminability'],
            explanation="",
        )
    
    return _score

