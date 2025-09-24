from inspect_ai import task, Task

from data_utils.load_mcqa_task import validate_and_summarize_dataset
from utils.setup import load_config
from enums import get_scorers_for_metrics


@task
def mcqa_dataset_score(dataset_path: str | None = None, metrics: str | None = None, run_name: str | None = None):
    """Task that loads a dataset and attaches scorers based on metrics arg.

    Example CLI:
      uv run inspect eval endpoints/score_dataset.py@mcqa_dataset_score
    """
    from data_utils.load_mcqa_task import load_mcqa_dataset
    from solvers.return_dataset import return_dataset

    # Read from config when not passed via -T
    if dataset_path is None or metrics is None or run_name is None:
        cfg = load_config()
        g = cfg.get("global", {})
        dataset_path = dataset_path or g.get("dataset")
        metric_list = metrics or g.get("metrics", [])
        run_name = run_name or g.get("run_name")

    else:
        if "," in metrics:
            metric_list = [m.strip().lower() for m in metrics.split(",") if m.strip()]
        else:
            metric_list = [m for m in metrics.split() if m]

    validate_and_summarize_dataset(dataset_path)

    scorers = get_scorers_for_metrics(metric_list)
    dataset = load_mcqa_dataset(dataset_path)

    return Task(
        dataset=dataset,
        solver=return_dataset(),
        scorer=scorers,
    )