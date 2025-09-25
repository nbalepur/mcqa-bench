from inspect_ai import task, Task, eval
from data_utils.return_dataset import return_dataset
from data_utils.merge_datasets import merge_and_shuffle_datasets
from utils.setup import load_config
from utils.enums import get_scorer_for_metric, Metrics
from model_utils.irt import IRTModel
import os


@task
def mcqa_skills(datasets: str | None = None, models: str | None = None, use_cot: bool | None = None, run_name: str | None = None, sample_to_score: dict = None, use_irt: bool = False):
    """Task that takes multiple anchor datasets, shuffles them evenly, and runs evaluation to get model skills."""
    
    cfg = load_config()
    skills_cfg = cfg.get("skills", {})
    
    if datasets:
        dataset_paths = [d.strip() for d in datasets.split(",") if d.strip()]
    else:
        dataset_paths = skills_cfg.get("datasets", [])
    if not dataset_paths:
        raise ValueError("Datasets are required")
      
    use_cot = use_cot if use_cot is not None else skills_cfg.get("use_cot", False)
    
    merged_samples = merge_and_shuffle_datasets(dataset_paths)
    
    # Choose metric and parameters based on whether IRT is being used
    scorer = get_scorer_for_metric(Metrics.DIFFICULTY.value, cfg, save_skills=False, sample_to_score=sample_to_score)
    if not use_irt:
        scorer = scorer[0]

    return Task(
        dataset=merged_samples,
        solver=return_dataset(),
        scorer=scorer,
    )


def run_skills_eval(datasets: str | None = None, models: str | None = None, use_cot: bool | None = None, run_name: str | None = None, limit: int = 5):
    """Run skills evaluation using eval() function."""
    cfg = load_config()
    skills_cfg = cfg.get("skills", {})
    global_cfg = cfg.get("global", {})
    
    if models:
        model_list = [m.strip() for m in models.split(",") if m.strip()]
    else:
        model_list = skills_cfg.get("models", [])
    if not model_list:
        raise ValueError("Models are required")
    
    eval_model = model_list[0]

    # Resolve run_name and use_cot from CLI -> skills config -> global config
    resolved_run_name = run_name or skills_cfg.get("run_name") or global_cfg.get("run_name")
    resolved_use_cot = use_cot if use_cot is not None else skills_cfg.get("use_cot", False)

    # Prepare task_args common fields
    task_args_common = {
        "datasets": datasets if datasets is not None else None,
        "use_cot": resolved_use_cot,
        "run_name": resolved_run_name,
    }   
    
    # Get model accuracy
    os.environ["INSPECT_EVAL_LOG_FILE_PATTERN"] = f"{resolved_run_name}_mcqa_skills"
    eval_logs = eval(
        "endpoints/run_skills.py@mcqa_skills",
        model=eval_model,
        limit=limit,
        task_args={
            **{k: v for k, v in task_args_common.items() if v is not None},
            "use_irt": False,
        },
    )
    
    # 2) Run IRT
    irt_model = IRTModel(eval_logs=eval_logs)
    irt_model.train()
    sample_to_score = irt_model.save(skills_cfg.get('skill_file', 'skills') + f'/{resolved_run_name}/')
    irt_model.clean_up()
    
    # 3) Add IRT scores
    os.environ["INSPECT_EVAL_LOG_FILE_PATTERN"] = f"{resolved_run_name}_mcqa_skills_with_irt"
    eval(
        "endpoints/run_skills.py@mcqa_skills",
        model=eval_model,
        limit=limit,
        task_args={
            **{k: v for k, v in task_args_common.items() if v is not None},
            'use_irt': True,
            'sample_to_score': sample_to_score,
        },
    )

def main():
    """Main entrypoint for running skills evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run skills evaluation")
    parser.add_argument("--datasets", type=str, help="Comma-separated list of dataset paths")
    parser.add_argument("--models", type=str, help="Comma-separated list of models")
    parser.add_argument("--use-cot", action="store_true", help="Use chain-of-thought prompting")
    parser.add_argument("--run-name", type=str, help="Name for this run")
    parser.add_argument("--limit", type=int, default=5, help="Number of samples to evaluate")
    
    args = parser.parse_args()
    
    run_skills_eval(
        datasets=args.datasets,
        models=args.models,
        use_cot=args.use_cot,
        run_name=args.run_name,
        limit=args.limit
    )


if __name__ == "__main__":
    main()
