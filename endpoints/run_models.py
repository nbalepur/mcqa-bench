from inspect_ai import task, Task
from inspect_ai.solver import multiple_choice
from inspect_ai.scorer import choice
from prompts.run_mcqa_prompts import load_mcqa_template

import subprocess
import os
import time
from dotenv import load_dotenv
load_dotenv()

from data_utils.load_mcqa_task import load_mcqa_dataset, validate_and_summarize_dataset

@task
def mcqa_task(dataset_path: str, use_cot: bool = False, with_question: bool = True):
    task_dataset = load_mcqa_dataset(dataset_path)
    return Task(dataset=task_dataset,
    solver=multiple_choice(template=load_mcqa_template(use_cot, with_question)),
    scorer=choice())

def _run_inspect(dataset_path: str, model: str, use_cot: bool = False, with_question: bool = True, **kwargs):
    """Build the command for running a single model."""
    cmd = [
        "uv", "run", "inspect", "eval", "endpoints/run_models.py@mcqa_task",
        "-T", f"dataset_path={dataset_path}",
        "-T", f"use_cot={use_cot}",
        "-T", f"with_question={with_question}",
        "--model", model
    ]
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    return cmd

def run_models(dataset_path: str, models: list, run_name: str = None, parallel: bool = False, use_cot: bool = False, with_question: bool = True, **kwargs):
    """Run MCQA evaluation for multiple models."""
    base_folder = run_name
    
    if parallel:
        processes = []
        
        for model in models:
            suffix = f'cot={use_cot}_with_question={with_question}'
            log_pattern = f"{base_folder}_{model.replace('/', '-')}_{suffix}"
            env = os.environ.copy()
            env["INSPECT_EVAL_LOG_FILE_PATTERN"] = log_pattern
            
            cmd = _run_inspect(dataset_path, model, **kwargs)
            
            # Run silently in background - results will be viewable on website
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            processes.append((model, process))
            time.sleep(0.5)
        
        for model, process in processes:
            process.wait()
        
    else:
        for model in models:
            log_pattern = f"{base_folder}_{model.replace('/', '-')}"
            os.environ["INSPECT_EVAL_LOG_FILE_PATTERN"] = log_pattern
            
            cmd = _run_inspect(dataset_path, model, **kwargs)
            subprocess.run(cmd, check=True, env=os.environ.copy())
