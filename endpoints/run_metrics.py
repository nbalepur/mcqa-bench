from inspect_ai import task, Task, eval
from data_utils.return_dataset import return_dataset
from prompts.run_mcqa_prompts import load_mcqa_template

from data_utils.load_mcqa_task import validate_and_summarize_dataset, load_mcqa_dataset
from utils.setup import load_config
from utils.enums import get_scorers_for_metrics


@task
def mcqa_metrics(dataset_path: str | None = None, metrics: str | None = None, run_name: str | None = None, model: str | None = None):
    """Task that loads a dataset and runs metrics with all specified parameters."""
    
    cfg = load_config()
    g = cfg.get("global", {})
    
    dataset_path = dataset_path or g.get("dataset")
    
    if metrics:
        metric_list = [m.strip().lower() for m in metrics.split(",") if m.strip()]
    else:
        metric_list = g.get("metrics", [])
    
    validate_and_summarize_dataset(dataset_path)
    scorers = get_scorers_for_metrics(metric_list, cfg, save_skills=False)
    dataset = load_mcqa_dataset(dataset_path)
    
    return Task(dataset=dataset,
    solver=return_dataset(),
    scorer=scorers)


def run_metrics_eval(dataset_path: str | None = None, metrics: str | None = None, run_name: str | None = None, model: str | None = None, limit: int = 5):
    """Run metrics evaluation using eval() function."""
    cfg = load_config()
    g = cfg.get("global", {})
    
    dataset_path = dataset_path or g.get("dataset")
    model = model or "openai/gpt-4o-mini"
    
    return eval(
        "endpoints/run_metrics.py@mcqa_metrics",
        model=model,
        limit=limit
    )
