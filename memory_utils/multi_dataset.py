from data_utils import p3
import random as random_
import pickle
import numpy as np
import torch
from scipy.special import softmax

np_rng = np.random.default_rng()

class MultiTaskMemory:
    def __init__(self, config, task_names, datasets, tokenizer, collator, random_seed=0):
        self.config = config
        self.task_names = task_names
        self.tokenizer = tokenizer
        self.datasets = datasets
        self.collator = collator
        self.random = random_.Random(random_seed)

    def add_dataset(self, task_name, ds):
        self.task_names.append(task_name)
        self.datasets.append(ds)

    def encode_examples(self, examples):
        inputs, labels = [], []
        for example in examples:
            inputs.append(example['original_input'])
            labels.append(example['original_answers'])
        input_encoding = self.tokenizer.batch_encode_plus(inputs, padding='max_length', truncation=True,
                                                          max_length=self.config.max_input_length)
        label_encoding = self.tokenizer.batch_encode_plus(labels, padding='max_length', truncation=True,
                                                          max_length=self.config.max_output_length)
        encoded_examples = []
        for b in range(len(examples)):
            encoded_examples.append({
                'input_ids': input_encoding['input_ids'][b],
                'attention_mask': input_encoding['attention_mask'][b],
                'labels': label_encoding['input_ids'][b],
                'task_name': examples[b]['task_name'],
                'original_input': examples[b]['original_input'],
                'original_answers': examples[b]['original_answers']
            })
        return encoded_examples

    def random_sample(self, k):
        ds_lens = [len(x) for x in self.datasets]
        sum_len = sum(ds_lens)
        sample_idxs = self.random.sample(range(sum_len), k)
        sample_idxs.sort()

        examples = []
        tasks = []

        j = 0
        offset = 0
        for idx in sample_idxs:
            while idx >= offset + ds_lens[j]:
                offset += ds_lens[j]
                j += 1
            tasks.append(self.task_names[j])
            examples.append(self.datasets[j][idx - offset])

        encoded_examples = self.encode_examples(examples)
        batch = self.collator(encoded_examples)
        return batch, tasks


class DatasetMemory:
    def __init__(self, ds, collator, random_seed=0):
        self.ds = ds
        self.collator = collator
        self.random = random_.Random(random_seed)

    def random_sample(self, k):
        if k < len(self.ds):
            sample_idxs = self.random.sample(range(len(self.ds)), k)
        else:
            sample_idxs = [_ for _ in range(len(self.ds))]
        examples = []
        tasks = []
        for idx in sample_idxs:
            examples.append(self.ds[idx])
            tasks.append(self.ds[idx]['task_name'])
        batch = self.collator(examples)
        return batch, tasks, sample_idxs

    def random_sample_from_indices_with_filling(self, k, indices, replayed_idxs=None, no_repeat=False):
        if no_repeat:
            indices = [x for x in indices if x not in replayed_idxs]
        if k > len(indices):
            indices = indices + self.random.sample(range(len(self.ds)), k - len(indices))

        sample_idxs = self.random.sample(indices, k)
        examples = []
        tasks = []
        for idx in sample_idxs:
            examples.append(self.ds[idx])
            tasks.append(self.ds[idx]['task_name'])
        batch = self.collator(examples)
        return batch, tasks, sample_idxs

    def weight_random_sampling(self, k, pred_forgets, weight_temp):
        weight = softmax(pred_forgets / weight_temp)
        sampled_idxs = np_rng.choice(len(self.ds), size=k, replace=False, p=weight)

        examples = []
        tasks = []
        for idx in sampled_idxs:
            examples.append(self.ds[idx])
            tasks.append(self.ds[idx]['task_name'])
        batch = self.collator(examples)
        return batch, tasks, sampled_idxs

class CoresetMemory:
    def __init__(self, ds, tasks, collator, coreset_info, random_seed=0):
        self.ds = ds
        self.collator = collator
        self.random = random_.Random(random_seed)

        self.indices = []
        idx = 0
        for task in tasks:
            for mat in coreset_info[task]:
                mat = torch.from_numpy(mat)
                mat_zd = mat.masked_fill(torch.eye(mat.shape[0]).bool(), 0.)
                mat_diags = mat.masked_select(torch.eye(mat.shape[0]).bool())
                scores = []
                for i in range(mat.shape[0]):
                    sim_score = mat[i].mean() / torch.sqrt(mat_diags.mean())
                    div_score = (mat_zd[i] / torch.sqrt(mat_diags)).sum() / (mat.shape[0] - 1)
                    cs_score = sim_score - div_score
                    scores.append(cs_score)
                argmax_i = np.argmax(scores)
                self.indices.append(idx + argmax_i)
                idx += mat.shape[0]

    def random_sample(self, k):
        if k < len(self.indices):
            sample_idxs = self.random.sample(self.indices, k)
        else:
            sample_idxs = self.indices
        examples = []
        tasks = []
        for idx in sample_idxs:
            examples.append(self.ds[idx])
            tasks.append(self.ds[idx]['task_name'])
        batch = self.collator(examples)
        return batch, tasks, sample_idxs




