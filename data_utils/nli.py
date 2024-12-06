import os
from torch.utils.data import Dataset
import json
import numpy as np

import logging
from sklearn.metrics import f1_score, accuracy_score
import datasets
import random

def get_nli_id2label(ds_name):
    if ds_name == 'snli':
        return [0,1,2] # [entailment, neutral, contradiction]
    else:
        raise NotImplementedError

class Subset(Dataset):
    def __init__(self, ds, idxs):
        super().__init__()
        self.ds = ds
        self.idxs = idxs
        if max(idxs) >= len(ds):
            raise ValueError('max(idxs) {} larger than size of ds {}'.format(max(idxs), len(ds)))
        print(max(idxs), len(ds))

    def __getitem__(self, idx):
        real_idx = self.idxs[idx]
        return self.ds[real_idx]

    def __len__(self):
        return len(self.idxs)

def partition_dataset(dataset):
    idxs = [_ for _ in range(len(dataset))]
    random.Random(0).shuffle(idxs)
    part0, part1 = idxs[:len(idxs) // 2], idxs[len(idxs) // 2:]
    ds0, ds1 = Subset(dataset, part0), Subset(dataset, part1)
    return ds0, ds1

class NLIDataManager:
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

    def update_model_config(self, config):
        labels = get_nli_id2label(self.config.dataset_name)
        self.config.num_labels = config.num_labels = len(labels)
        self.config.label2id = config.label2id = {v: i for i,v in enumerate(labels)}
        self.config.id2label = config.id2label = {i: label for label, i in config.label2id.items()}

    def load_dataset(self, config, partition=None):
        dataset_name = config.dataset_name
        split_datasets = datasets.load_dataset(dataset_name)

        def preprocess_function(examples):
            # Tokenize the texts
            text_pairs = [(a,b) for a,b in zip(examples['premise'], examples['hypothesis'])]
            result = self.tokenizer(text_pairs, max_length=max_seq_length, truncation=True)

            # Map labels to IDs (not necessary for GLUE tasks)
            result['label'] = examples['label']
            return result

        max_seq_length = config.max_seq_length

        for split, dataset in split_datasets.items():
            dataset = dataset.map(preprocess_function, batched=True)
            #if split != 'test':
            dataset = dataset.filter(lambda example: example['label'] != -1)
            split_datasets[split] = dataset

        ori_train_dataset = split_datasets['train']
        eval_dataset = split_datasets["validation"]
        test_dataset = split_datasets['test']

        pt_dataset, ocl_dataset = partition_dataset(ori_train_dataset)

        return pt_dataset, ocl_dataset, eval_dataset, test_dataset

    def get_metrics_func(self):
        def compute_metrics(p):
            result = {}
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions # [N, L]
            labels = p.label_ids
            pred_labels = preds.argmax(-1)
            f1 = f1_score(labels, pred_labels, average='macro')
            acc = accuracy_score(labels, pred_labels)
            result["macro_f1"] = f1
            result["acc"] = acc
            return result
        return compute_metrics





