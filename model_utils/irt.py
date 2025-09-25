from inspect_ai.log import EvalLog
from inspect_ai.scorer import CORRECT

import os
import json
from collections import defaultdict
from contextlib import redirect_stdout, redirect_stderr

from py_irt.dataset import Dataset
from py_irt.config import IrtConfig
from py_irt.training import IrtModelTrainer


class IRTModel:

    def __init__(self, eval_logs: list[EvalLog]):

        self.dataset = self.convert_to_dataset(eval_logs)
        self.config = IrtConfig(model_type='2pl', log_every=1000, dropout=.2, seed=42)
        self.trainer = IrtModelTrainer(config=self.config, data_path=None, dataset=self.dataset, verbose=True)

    def convert_to_dataset(self, eval_logs: list[EvalLog]):

        subject_to_responses = defaultdict(lambda: defaultdict(int))
        sample_to_score = dict()
        for eval_log in eval_logs:
            for eval_sample in eval_log.samples:
                key = list(eval_sample.scores.keys())[0]
                eval_scores = eval_sample.scores[key].metadata['scores']
                for eval_score in eval_scores:
                    subject_to_responses[eval_score['model']][f"q{eval_sample.id}"] = int(eval_score['score'] == CORRECT)
                    curr_data = sample_to_score.get(eval_sample.id, {'accuracy': []})
                    curr_data['accuracy'].append(eval_score)
                    sample_to_score[eval_sample.id] = curr_data
        self.sample_to_score = sample_to_score

        self.temp_file = 'tmp/irt_dataset.jsonl'
        os.makedirs(os.path.dirname(self.temp_file), exist_ok=True)
        with open(self.temp_file, 'w') as f:
            for subject_id, subject_data in subject_to_responses.items():
                f.write(json.dumps({"subject_id": subject_id, "responses": {k: v for k, v in sorted(subject_data.items(), key=lambda x: int(x[0][1:]))}}) + '\n')
        return Dataset.from_jsonlines(self.temp_file)

    def train(self, epochs: int = 5000, device: str = 'cpu'):
        self.trainer.train(epochs=epochs, device=device)

    def save(self, path: str):
        params = self.trainer.last_params
        json_params = {}
        for key, value in params.items():
            if hasattr(value, 'tolist'):
                json_params[key] = value.tolist()
            else:
                json_params[key] = value

        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'irt_params.json'), 'w') as f:
            json.dump(json_params, f, indent=2)

        item_id_to_sample = json_params["item_ids"]
        for idx, (diff, disc) in enumerate(zip(json_params['diff'], json_params['disc'])):
            self.sample_to_score[int(item_id_to_sample[idx][1:])]['difficulty'] = diff
            self.sample_to_score[int(item_id_to_sample[idx][1:])]['discriminability'] = disc

        return self.sample_to_score

    def clean_up(self):
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)
        temp_dir = os.path.dirname(self.temp_file)
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)