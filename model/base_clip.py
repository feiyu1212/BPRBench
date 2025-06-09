import sys
import json
from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.clip_utils import FilteredImageFolder, convert_df
sys.path.append('CONCH/')
from conch.downstream.zeroshot_path import zero_shot_classifier, run_zeroshot
# from conch.downstream.zeroshot_path2 import zero_shot_classifier, run_zeroshot


class BaseClip(ABC):
    def __init__(self, ckpt):
        super().__init__()
        self._init_model(ckpt)
        self._load_prompts()

    @abstractmethod
    def _init_model(self, ckpt):
        pass

    def _load_prompts(self):
        with open('../data/templates.json', 'r') as f:
            self.templates = json.load(f)['templates']

        # with open('../data/question_with_prompt.json', 'r') as f:
        #     self.task_prompts = json.load(f)

        with open('../data/question_with_prompt.json', 'r') as f:
            self.task_prompts = json.load(f)

    def _get_task_weights(self, i):
        # task_prompts = {k: v for _, vs in self.task_prompts[i].items() for k, v in vs.items()}
        task_prompts = {k: v for k, vs in self.task_prompts[i]['prompt'].items() for k, v in vs.items()}
        diseases = {task: prompts for task, prompts in task_prompts.items()}
        classnames = [classname for classname, prompts in diseases.items()]
        classnames_text = [prompts + [classname] for classname, prompts in diseases.items()]
        id_to_class = {i: classname for i, (classname, prompts) in enumerate(diseases.items())}
        task_weights = zero_shot_classifier(self.model, classnames_text, self.templates, tokenizer=self.tokenizer, device=self.device)
        return task_weights, classnames, classnames_text, id_to_class

    def predict_df(self, data_source, i, allowed_classes=None):
        with torch.cuda.amp.autocast(), torch.no_grad():
            task_weights, classnames, classnames_text, id_to_class = self._get_task_weights(i)
            dataset = FilteredImageFolder(data_source, transform=self.transform, allowed_classes=allowed_classes)
            dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8)
            results, dump = run_zeroshot(self.model, task_weights, dataloader, self.device,
                                         dump_results=True, metrics=['bacc', 'weighted_f1'])
            return convert_df(dump, dataset, classnames)



